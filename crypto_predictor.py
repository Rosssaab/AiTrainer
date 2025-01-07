import os
from datetime import datetime
from config import (
    DB_CONNECTION_STRING,
    DB_SERVER,
    DB_NAME,
    DB_USER,
    DB_PASSWORD
)
import logging
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import glob
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class CryptoPredictor:
    def __init__(self):
        # Set up enhanced logging
        self.setup_logging()
        
        self.log.info("=== Starting Crypto Predictor ===")
        self.log.info(f"Database: {DB_NAME} on {DB_SERVER}")
        
        # Add model path checking
        self.model_base_path = os.path.join(os.path.dirname(__file__), 'models')
        if not os.path.exists(self.model_base_path):
            os.makedirs(self.model_base_path)
            self.log.info(f"Created models directory at {self.model_base_path}")
        else:
            self.log.info(f"Using existing models directory at {self.model_base_path}")
        
        # Use the connection string directly from config
        try:
            # Change this part - use DB_CONNECTION_STRING directly
            self.engine = create_engine(DB_CONNECTION_STRING)
            
            # Test connection
            with self.engine.connect() as conn:
                self.log.info("Database connection successful")
        except Exception as e:
            self.log.error(f"Database connection error: {str(e)}")
            raise

        self.sequence_length = 10
        self.confidence_threshold = 0.7
        
        # Create directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        self.stablecoin_identifiers = [
            'usdt', 'usdc', 'dai', 'usde', 'usd0', 'fdusd', 'usds',
            'aammdai',
            'aave-amm-unidaiusdc'
        ]
        self.log.info(f"Initialized with {len(self.stablecoin_identifiers)} stablecoins filtered")
        
        self.batch_id = None  # Will be set when predictions start

    def setup_logging(self):
        """Set up enhanced logging configuration"""
        try:
            # Ensure logs directory exists with full path
            log_dir = os.path.join(os.path.dirname(__file__), 'logs')
            os.makedirs(log_dir, exist_ok=True)
            
            log_date = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f"Crypto_predictor_{log_date}.log")
            
            # Test write permissions
            with open(log_file, 'a') as f:
                f.write('')
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s\n'
                'Function: %(funcName)s - Line: %(lineno)d\n'
                '-------------------'
            )
            
            # File handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            
            # Setup logger
            self.log = logging.getLogger('CryptoPredictor')
            self.log.setLevel(logging.DEBUG)
            self.log.addHandler(file_handler)
            self.log.addHandler(console_handler)
            
            self.log.info(f"Log file created at: {log_file}")
        except Exception as e:
            print(f"Error setting up logging: {str(e)}")
            raise

    def get_latest_data(self, crypto_id):
        self.log.debug(f"Fetching latest data for {crypto_id}")
        query = text("""
            WITH LatestData AS (
                SELECT TOP 10
                    d.crypto_id,
                    d.price_date,
                    d.current_price,
                    d.market_cap,
                    d.total_volume,
                    d.price_change_24h,
                    s.sentiment_votes_up,
                    s.sentiment_votes_down,
                    s.public_interest_score
                FROM coingecko_crypto_daily_data d
                LEFT JOIN coingecko_crypto_sentiment s 
                    ON d.crypto_id = s.crypto_id 
                    AND CAST(d.price_date AS DATE) = CAST(s.metric_date AS DATE)
                WHERE d.crypto_id = :crypto_id
                ORDER BY d.price_date DESC
            )
            SELECT * FROM LatestData ORDER BY price_date ASC
        """)
        
        with self.engine.connect() as connection:
            data = pd.read_sql(query, connection, params={"crypto_id": crypto_id})
            self.log.debug(f"Retrieved {len(data)} rows of data")
            self.log.debug(f"Data columns: {data.columns.tolist()}")
            self.log.debug(f"Latest price date: {data['price_date'].max()}")
            return data

    def prepare_data(self, data):
        self.log.debug("Preparing data for prediction")
        try:
            # Get the last current price for later use
            self.last_current_price = float(data['current_price'].iloc[-1])
            self.log.debug(f"Last current price: {self.last_current_price}")
            
            # Log data statistics
            self.log.debug("\nData Statistics:")
            self.log.debug(data.describe())
            
            # Log data shape and missing values
            self.log.debug(f"Data shape before preparation: {data.shape}")
            self.log.debug("Missing values:\n" + str(data.isnull().sum()))
            
            # Prepare features for prediction
            features = ['current_price', 'market_cap', 'total_volume', 'price_change_24h',
                       'sentiment_votes_up', 'sentiment_votes_down', 'public_interest_score']
            X = data[features].fillna(0).values
            
            # Scale the data
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)
            
            # Reshape for LSTM (samples, time steps, features)
            X = X.reshape(1, X.shape[0], X.shape[1])
            
            self.log.debug(f"X shape after scaling: {X.shape}")
            self.log.debug(f"X sample:\n{X[0][:2]}")  # Show first two time steps
            
            return X
            
        except Exception as e:
            self.log.error(f"Error in prepare_data: {str(e)}")
            raise

    def save_prediction(self, crypto_id, predictions, confidence, model_version):
        self.log.debug(f"Saving predictions for {crypto_id}")
        try:
            # Check for NaN values
            if np.isnan(predictions).any():
                self.log.error(f"NaN predictions detected for {crypto_id}")
                return False
            
            # Use last_current_price instead of last_price
            current_price = float(self.last_current_price)  # Convert to native float
            
            # Calculate predicted prices
            price_24h = float(current_price * (1 + predictions[0][0]))
            price_48h = float(current_price * (1 + predictions[0][0] * 1.5))
            price_3d = float(current_price * (1 + predictions[0][0] * 1.75))
            price_7d = float(current_price * (1 + predictions[0][0] * 2))
            
            prediction_data = {
                'crypto_id': crypto_id,
                'prediction_date': datetime.now(),
                'prediction_created_at': datetime.now(),
                'price_24h': price_24h,
                'price_48h': price_48h,
                'price_3d': price_3d,
                'price_7d': price_7d,
                'confidence_score': float(confidence),
                'model_version': model_version,
                'batch_id': self.batch_id,
                'current_price': current_price
            }
            
            with self.engine.connect() as connection:
                insert_query = text("""
                    INSERT INTO coingecko_crypto_predictions (
                        crypto_id, prediction_date, prediction_created_at,
                        price_24h, price_48h, price_3d, price_7d,
                        confidence_score, model_version, batch_id, current_price
                    ) VALUES (
                        :crypto_id, :prediction_date, :prediction_created_at,
                        :price_24h, :price_48h, :price_3d, :price_7d,
                        :confidence_score, :model_version, :batch_id, :current_price
                    )
                """)
                
                connection.execute(insert_query, prediction_data)
                connection.commit()
                self.log.info(f"Successfully saved predictions for {crypto_id}")
                
        except Exception as e:
            self.log.error(f"Error saving predictions for {crypto_id}: {str(e)}")
            self.log.error(f"Predictions shape: {predictions.shape}")
            self.log.error(f"Predictions content: {predictions}")
            raise

    def get_next_batch_id(self):
        """Get the next available batch ID from the database"""
        try:
            with self.engine.connect() as conn:
                query = text("""
                    SELECT ISNULL(MAX(batch_id), 0) + 1 
                    FROM coingecko_crypto_predictions
                """)
                result = conn.execute(query).scalar()
                return result
        except Exception as e:
            self.log.error(f"Error getting next batch ID: {str(e)}")
            return 1  # Fallback to 1 if query fails

    def make_predictions(self):
        try:
            # Get next batch ID at the start
            self.batch_id = self.get_next_batch_id()
            self.log.info(f"Starting prediction batch {self.batch_id}")
            
            self.log.info("Starting prediction process")
            processed_count = 0
            failed_count = 0
            no_model_count = 0
            
            # Modified query to exclude test coins
            stablecoin_list = "'" + "','".join(self.stablecoin_identifiers) + "'"
            query = text(f"""
                SELECT id, symbol, name, market_cap_rank 
                FROM coingecko_crypto_master 
                WHERE id NOT IN ({stablecoin_list})
                    AND market_cap_rank IS NOT NULL
                    AND id NOT LIKE 'test-%'           -- Exclude test coins
                    AND id NOT LIKE 'testnet%'         -- Exclude testnet coins
                    AND market_cap_rank <= 100         -- Limit to top 100
                ORDER BY market_cap_rank ASC
            """)
            
            with self.engine.connect() as connection:
                result = connection.execute(query)
                crypto_ids = result.fetchall()
                self.log.info(f"Found {len(crypto_ids)} cryptocurrencies to process")
            
            for crypto_id, symbol, name, rank in crypto_ids:
                try:
                    self.log.info(f"\nProcessing {name} ({crypto_id}) - Rank {rank}")
                    
                    # Look for model with more detailed logging
                    model_pattern = os.path.join(self.model_base_path, f"{crypto_id}_*.keras")
                    model_files = glob.glob(model_pattern)
                    
                    if not model_files:
                        self.log.warning(f"No model found for {name} (looking in {model_pattern})")
                        no_model_count += 1
                        continue
                        
                    latest_model_file = max(model_files, key=os.path.getctime)
                    self.log.info(f"Using model: {os.path.basename(latest_model_file)}")
                    
                    try:
                        # More robust model loading with custom objects
                        custom_objects = None
                        model = load_model(latest_model_file, 
                                         compile=False, 
                                         custom_objects=custom_objects)
                        
                        # Basic compilation settings
                        model.compile(optimizer='adam', 
                                    loss='mse',
                                    metrics=['mae'])
                                    
                    except Exception as model_error:
                        self.log.error(f"Error loading model for {name}: {str(model_error)}")
                        self.log.error(f"Model file: {latest_model_file}")
                        failed_count += 1
                        continue

                    data = self.get_latest_data(crypto_id)
                    if len(data) < self.sequence_length:
                        self.log.warning(f"Insufficient data for {name}")
                        continue
                        
                    X = self.prepare_data(data)
                    
                    # Add error handling for predictions
                    try:
                        predictions = model.predict(X, verbose=0)
                    except Exception as pred_error:
                        self.log.error(f"Prediction error for {name}: {str(pred_error)}")
                        failed_count += 1
                        continue
                    
                    self.log.debug(f"Raw prediction output: {predictions}")
                    self.save_prediction(crypto_id, predictions, 0.8, 
                                       os.path.basename(latest_model_file).replace('.keras', ''))
                    
                    processed_count += 1
                    
                except Exception as e:
                    self.log.error(f"Error processing {name}: {str(e)}")
                    failed_count += 1
                    continue
                
            # Add summary statistics
            self.log.info("\n=== Prediction Summary ===")
            self.log.info(f"Total cryptocurrencies: {len(crypto_ids)}")
            self.log.info(f"Successfully processed: {processed_count}")
            self.log.info(f"Failed to process: {failed_count}")
            self.log.info(f"No models found: {no_model_count}")
            self.log.info("========================")
            
        except Exception as e:
            self.log.error(f"Error in make_predictions: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        predictor = CryptoPredictor()
        predictor.make_predictions()
    except Exception as e:
        logging.error(f"Main execution error: {str(e)}")