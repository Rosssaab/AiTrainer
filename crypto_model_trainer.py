import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sqlalchemy import create_engine, text
import os
from datetime import datetime
from config import DB_CONNECTION_STRING
import warnings
import glob
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import gc
import tensorflow as tf
import logging
from tensorflow.keras import backend as K

# Add this to suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')

warnings.filterwarnings('ignore', category=UserWarning)

class CryptoModelTrainer:
    def __init__(self):
        # Set up enhanced logging first
        self.setup_logging()
        
        self.log.info("=== Starting Crypto Model Trainer ===")
        
        # Use the connection string directly from config
        try:
            self.engine = create_engine(DB_CONNECTION_STRING)
            
            # Test connection
            with self.engine.connect() as conn:
                self.log.info("Database connection successful")
        except Exception as e:
            self.log.error(f"Database connection error: {str(e)}")
            raise

        self.min_days_required = 14
        self.sequence_length = 5
        self.models_dir = os.path.join(os.path.dirname(__file__), 'models')
        
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
            self.log.info(f"Created models directory at {self.models_dir}")

        self.batch_id = None

    def setup_logging(self):
        """Set up enhanced logging configuration"""
        try:
            # Ensure logs directory exists
            log_dir = os.path.join(os.path.dirname(__file__), 'logs')
            os.makedirs(log_dir, exist_ok=True)
            
            log_date = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f"Crypto_trainer_{log_date}.log")
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s\n'
                'Function: %(funcName)s - Line: %(lineno)d\n'
                '-------------------'
            )
            
            # File handler with immediate flush
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            file_handler.flush = True
            
            # Console handler with immediate flush
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            console_handler.flush = True
            
            # Setup logger
            self.log = logging.getLogger('CryptoTrainer')
            self.log.setLevel(logging.INFO)
            self.log.addHandler(file_handler)
            self.log.addHandler(console_handler)
            
            # Force immediate output
            self.log.propagate = False
            
            # Disable output buffering
            import sys
            sys.stdout.reconfigure(line_buffering=True)
            
            self.log.info(f"Log file created at: {log_file}")
        except Exception as e:
            print(f"Error setting up logging: {str(e)}")
            raise

    def prepare_data(self, data, target_column):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        
        # Drop rows where current_price is 0 or null to avoid division by zero
        data = data[data['current_price'].notna() & (data['current_price'] != 0)].copy()
        
        # Calculate percentage changes instead of raw values
        for col in ['current_price', 'market_cap', 'total_volume']:
            data[f'{col}_pct_change'] = data[col].pct_change().fillna(0).clip(-0.5, 0.5)
        
        # Use percentage changes for training
        training_features = [
            'current_price_pct_change', 
            'market_cap_pct_change',
            'total_volume_pct_change',
            'price_change_24h',
            'sentiment_votes_up',
            'sentiment_votes_down',
            'public_interest_score'
        ]
        
        scaled_data = scaler.fit_transform(data[training_features])
        
        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length - target_column):
            X.append(scaled_data[i:(i + self.sequence_length)])
            current = data['current_price'].iloc[i + self.sequence_length]
            future = data['current_price'].iloc[i + self.sequence_length + target_column]
            
            # Add safety checks for percentage change calculation
            if current > 0 and not np.isnan(current) and not np.isnan(future):
                pct_change = ((future - current) / current)
                pct_change = np.clip(pct_change, -0.5, 0.5)
                y.append(pct_change)
            else:
                y.append(0)  # Use 0 as fallback for invalid cases
            
        return np.array(X), np.array(y), scaler

    def create_model(self, input_shape):
        model = Sequential([
            Input(shape=input_shape),
            LSTM(50, return_sequences=True),
            Dropout(0.3),
            LSTM(50),
            Dropout(0.3),
            Dense(25, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='tanh')
        ])
        
        optimizer = Adam(learning_rate=0.0005)
        model.compile(optimizer=optimizer, loss='huber')
        return model

    def cleanup_old_models(self):
        """Remove all previous model files before starting new training"""
        try:
            model_files = glob.glob(os.path.join(self.models_dir, '*.keras'))
            scaler_files = glob.glob(os.path.join(self.models_dir, '*_scaler.pkl'))
            
            for file_path in model_files + scaler_files:
                try:
                    os.remove(file_path)
                    self.log.info(f"Removed old file: {os.path.basename(file_path)}")
                except Exception as e:
                    self.log.error(f"Error removing {file_path}: {str(e)}")
                    
            self.log.info("Cleaned up old model files")
        except Exception as e:
            self.log.error(f"Error during model cleanup: {str(e)}")

    def train_model_with_timeout(self, model, X, y, early_stopping, timeout_seconds=300):
        """Train model with timeout"""
        done = threading.Event()
        result = {'history': None, 'error': None}
        
        def training_target():
            try:
                result['history'] = model.fit(
                    X, y,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stopping],
                    verbose=0
                )
            except Exception as e:
                result['error'] = str(e)
            finally:
                done.set()

        thread = threading.Thread(target=training_target)
        thread.start()
        thread.join(timeout=timeout_seconds)
        
        if thread.is_alive():
            K.clear_session()
            thread.join(timeout=1)
            return None
            
        if result['error']:
            self.log.error(f"Training error: {result['error']}")
            return None
            
        return result['history']

    def get_next_batch_id(self):
        """Get the next available batch ID from the database"""
        try:
            with self.engine.connect() as conn:
                query = text("""
                    SELECT ISNULL(MAX(batch_id), 0) + 1 
                    FROM coingecko_model_performance
                """)
                result = conn.execute(query).scalar()
                return result
        except Exception as e:
            self.log.error(f"Error getting next batch ID: {str(e)}")
            return 1

    def train_models(self):
        try:
            self.batch_id = self.get_next_batch_id()
            self.log.info(f"Starting training batch {self.batch_id}")
            
            self.cleanup_old_models()
            start_time = datetime.now()

            # Initial query to get eligible cryptocurrencies
            query = text("""
                WITH RankedCryptos AS (
                    SELECT 
                        m.id, 
                        m.symbol, 
                        m.name, 
                        m.market_cap_rank,
                        COUNT(DISTINCT d.price_date) as day_count
                    FROM coingecko_crypto_master m
                    INNER JOIN coingecko_crypto_daily_data d 
                        ON m.id = d.crypto_id
                    GROUP BY 
                        m.id, 
                        m.symbol, 
                        m.name, 
                        m.market_cap_rank
                    HAVING COUNT(DISTINCT d.price_date) >= :min_days
                )
                SELECT * 
                FROM RankedCryptos
                WHERE market_cap_rank IS NOT NULL
                ORDER BY market_cap_rank ASC
            """)
            
            df = pd.read_sql(query, self.engine, params={'min_days': self.min_days_required})
            total_coins = len(df)
            total_models = total_coins * 4  # 4 timeframes per coin
            models_completed = 0
            
            self.log.info(f"Found {total_coins} eligible cryptocurrencies")
            
            for idx, row in df.iterrows():
                try:
                    coin_id = row['id']
                    name = row['name']
                    coin_start_time = datetime.now()
                    
                    # Calculate and display progress
                    progress = (idx + 1) / total_coins * 100
                    elapsed_time = (datetime.now() - start_time).total_seconds() / 60
                    eta_minutes = (elapsed_time / (idx + 1) * (total_coins - idx - 1)) if idx > 0 else 0
                    
                    self.log.info(f"\nProcessing {name} ({coin_id}) - {progress:.1f}% complete")
                    self.log.info(f"ETA: {eta_minutes:.1f} minutes remaining")
                    
                    # Get training data
                    data_query = text("""
                        SELECT 
                            d.price_date,
                            d.current_price,
                            d.market_cap,
                            d.total_volume,
                            d.price_change_24h,
                            COALESCE(s.sentiment_votes_up, 0) as sentiment_votes_up,
                            COALESCE(s.sentiment_votes_down, 0) as sentiment_votes_down,
                            COALESCE(s.public_interest_score, 0) as public_interest_score
                        FROM coingecko_crypto_daily_data d
                        LEFT JOIN coingecko_crypto_sentiment s 
                            ON d.crypto_id = s.crypto_id 
                            AND CAST(d.price_date AS DATE) = CAST(s.metric_date AS DATE)
                        WHERE d.crypto_id = :crypto_id
                        ORDER BY d.price_date ASC
                    """)
                    
                    data = pd.read_sql(data_query, self.engine, params={'crypto_id': coin_id})
                    
                    if data.empty:
                        self.log.warning(f"No data found for {name}")
                        continue
                        
                    # Train models for different prediction windows
                    for days in [1, 2, 3, 7]:  # 24h, 48h, 3d, 7d
                        try:
                            model_name = f"{coin_id}_{days}d"
                            self.log.info(f"Training {model_name} model")
                            
                            X, y, scaler = self.prepare_data(data, days)
                            
                            if len(X) < 50:
                                self.log.warning(f"Insufficient sequences for {model_name}")
                                continue
                            
                            K.clear_session()
                            
                            model = self.create_model((self.sequence_length, X.shape[2]))
                            early_stopping = EarlyStopping(
                                monitor='val_loss',
                                patience=5,
                                restore_best_weights=True
                            )
                            
                            history = self.train_model_with_timeout(model, X, y, early_stopping)
                            
                            if history is None:
                                self.log.warning(f"Training timed out for {model_name}")
                                continue
                            
                            model_path = os.path.join(self.models_dir, f"{model_name}.keras")
                            model.save(model_path)
                            
                            scaler_path = os.path.join(self.models_dir, f"{model_name}_scaler.pkl")
                            pd.to_pickle(scaler, scaler_path)
                            
                            val_loss = float(history.history['val_loss'][-1])
                            
                            mae_metrics = {
                                '24h': val_loss if days == 1 else None,
                                '48h': val_loss if days == 2 else None,
                                '3d': val_loss if days == 3 else None,
                                '7d': val_loss if days == 7 else None
                            }
                            
                            rmse_metrics = mae_metrics.copy()
                            
                            self.save_model_performance(
                                model_name, 
                                mae_metrics, 
                                rmse_metrics, 
                                len(X), 
                                f"Model trained for {coin_id} with {days}d prediction window"
                            )
                            
                            self.log.info(f"Saved {model_name} model and scaler (validation loss: {val_loss:.6f})")
                            models_completed += 1
                            
                            del model, history
                            gc.collect()
                            
                        except Exception as e:
                            self.log.error(f"Error training {model_name}: {str(e)}")
                            continue
                            
                    coin_duration = (datetime.now() - coin_start_time).total_seconds() / 60
                    self.log.info(f"Completed {name} in {coin_duration:.1f} minutes")
                    
                except Exception as e:
                    self.log.error(f"Error processing {name}: {str(e)}")
                    continue
                    
            total_duration = (datetime.now() - start_time).total_seconds() / 3600
            self.log.info(f"\nTraining completed in {total_duration:.1f} hours")
            self.log.info(f"Successfully trained {models_completed} out of {total_models} models")
            
        except Exception as e:
            self.log.error(f"Error in train_models: {str(e)}")
        finally:
            try:
                self.engine.dispose()
            except:
                pass
            gc.collect()

    def save_model_performance(self, model_version, mae_metrics, rmse_metrics, samples=0, notes=""):
        """Save model performance metrics to database"""
        try:
            mae_24h = float(mae_metrics.get('24h', 0)) if mae_metrics.get('24h') is not None and not np.isnan(mae_metrics.get('24h', 0)) else 0
            mae_48h = float(mae_metrics.get('48h', 0)) if mae_metrics.get('48h') is not None and not np.isnan(mae_metrics.get('48h', 0)) else 0
            mae_3d = float(mae_metrics.get('3d', 0)) if mae_metrics.get('3d') is not None and not np.isnan(mae_metrics.get('3d', 0)) else 0
            mae_7d = float(mae_metrics.get('7d', 0)) if mae_metrics.get('7d') is not None and not np.isnan(mae_metrics.get('7d', 0)) else 0
            
            rmse_24h = float(rmse_metrics.get('24h', 0)) if rmse_metrics.get('24h') is not None and not np.isnan(rmse_metrics.get('24h', 0)) else 0
            rmse_48h = float(rmse_metrics.get('48h', 0)) if rmse_metrics.get('48h') is not None and not np.isnan(rmse_metrics.get('48h', 0)) else 0
            rmse_3d = float(rmse_metrics.get('3d', 0)) if rmse_metrics.get('3d') is not None and not np.isnan(rmse_metrics.get('3d', 0)) else 0
            rmse_7d = float(rmse_metrics.get('7d', 0)) if rmse_metrics.get('7d') is not None and not np.isnan(rmse_metrics.get('7d', 0)) else 0

            query = text("""
                INSERT INTO coingecko_model_performance (
                    model_version, training_date,
                    mae_24h, mae_48h, mae_3d, mae_7d,
                    rmse_24h, rmse_48h, rmse_3d, rmse_7d,
                    training_samples, notes, batch_id
                ) VALUES (
                    :model_version, :training_date,
                    :mae_24h, :mae_48h, :mae_3d, :mae_7d,
                    :rmse_24h, :rmse_48h, :rmse_3d, :rmse_7d,
                    :training_samples, :notes, :batch_id
                )
            """)
            
            with self.engine.connect() as conn:
                conn.execute(query, {
                    'model_version': model_version,
                    'training_date': datetime.now(),
                    'mae_24h': mae_24h,
                    'mae_48h': mae_48h,
                    'mae_3d': mae_3d,
                    'mae_7d': mae_7d,
                    'rmse_24h': rmse_24h,
                    'rmse_48h': rmse_48h,
                    'rmse_3d': rmse_3d,
                    'rmse_7d': rmse_7d,
                    'training_samples': int(samples) if samples is not None else 0,
                    'notes': notes if notes else '',
                    'batch_id': self.batch_id
                })
                conn.commit()
                
        except Exception as e:
            self.log.error(f"Error saving model performance: {str(e)}")

if __name__ == "__main__":
    try:
        trainer = CryptoModelTrainer()
        trainer.train_models()
    except Exception as e:
        print(f"Fatal error: {str(e)}")