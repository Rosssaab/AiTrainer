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
from config import DB_CONNECTION_STRING, DB_SERVER, DB_NAME, DB_USER, DB_PASSWORD
import warnings
import glob
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import gc
import tensorflow as tf

# Add this to suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')

warnings.filterwarnings('ignore', category=UserWarning)

class CryptoModelTrainer:
    def __init__(self):
        # Replace the current connection string construction with the imported one
        self.connection_str = (
            f'mssql+pyodbc:///?odbc_connect={DB_CONNECTION_STRING}'
        )
        
        try:
            self.engine = create_engine(self.connection_str)
            # Test connection
            with self.engine.connect() as conn:
                self.log("Database connection successful")
        except Exception as e:
            self.log(f"Database connection error: {str(e)}")
            raise

        self.min_days_required = 14
        self.sequence_length = 5
        self.models_dir = os.path.join(os.path.dirname(__file__), 'models')
        
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
            self.log(f"Created models directory at {self.models_dir}")

        self.batch_id = None  # Will be set when training starts

    def log(self, message):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

    def prepare_data(self, data, target_column):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        features = ['current_price', 'market_cap', 'total_volume', 'price_change_24h',
                   'sentiment_votes_up', 'sentiment_votes_down', 'public_interest_score']
        
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
            # Calculate target as percentage change
            current = data['current_price'].iloc[i + self.sequence_length]
            future = data['current_price'].iloc[i + self.sequence_length + target_column]
            pct_change = ((future - current) / current).clip(-0.5, 0.5)  # Clip extreme changes
            y.append(pct_change)
            
        return np.array(X), np.array(y), scaler

    def create_model(self, input_shape):
        model = Sequential([
            Input(shape=input_shape),
            LSTM(50, return_sequences=True),
            Dropout(0.3),  # Increased dropout
            LSTM(50),
            Dropout(0.3),  # Increased dropout
            Dense(25, activation='relu'),  # Added intermediate layer
            Dropout(0.2),  # Added dropout
            Dense(1, activation='tanh')  # Changed to tanh for bounded output
        ])
        
        # Reduced learning rate
        optimizer = Adam(learning_rate=0.0005)  # Reduced from 0.001
        model.compile(optimizer=optimizer, loss='huber')  # Changed to huber loss for robustness
        return model

    def cleanup_old_models(self):
        """Remove all previous model files before starting new training"""
        try:
            # Delete all .keras and .pkl files in models directory
            model_files = glob.glob(os.path.join(self.models_dir, '*.keras'))
            scaler_files = glob.glob(os.path.join(self.models_dir, '*_scaler.pkl'))
            
            for file_path in model_files + scaler_files:
                try:
                    os.remove(file_path)
                    self.log(f"Removed old file: {os.path.basename(file_path)}")
                except Exception as e:
                    self.log(f"Error removing {file_path}: {str(e)}")
                    
            self.log("Cleaned up old model files")
        except Exception as e:
            self.log(f"Error during model cleanup: {str(e)}")

    def train_model_with_timeout(self, model, X, y, early_stopping, timeout_seconds=300):
        """Train model with timeout using Event-based approach"""
        done = threading.Event()
        result = {'history': None, 'error': None}
        
        def training_target():
            try:
                result['history'] = model.fit(
                    X, y,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[
                        early_stopping,
                        tf.keras.callbacks.ReduceLROnPlateau(
                            monitor='val_loss',
                            factor=0.5,
                            patience=3,
                            min_lr=0.0001
                        )
                    ],
                    verbose=0
                )
            except Exception as e:
                result['error'] = str(e)
            finally:
                done.set()  # Always set the event when done

        training_thread = threading.Thread(target=training_target)
        training_thread.daemon = True  # Make thread daemon so it doesn't block program exit
        training_thread.start()
        
        # Wait for training to complete or timeout
        if not done.wait(timeout=timeout_seconds):
            self.log(f"Training timed out after {timeout_seconds} seconds")
            return None
        
        if result['error']:
            raise Exception(result['error'])
            
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
            self.log(f"Error getting next batch ID: {str(e)}")
            return 1  # Fallback to 1 if query fails

    def train_models(self):
        """Main training method"""
        try:
            # Get next batch ID at the start of training
            self.batch_id = self.get_next_batch_id()
            self.log(f"Starting training batch {self.batch_id}")
            
            self.cleanup_old_models()
            start_time = datetime.now()
            
            # Test database connection before proceeding
            try:
                with self.engine.connect() as connection:
                    result = connection.execute(text("SELECT @@VERSION"))
                    version = result.scalar()
                    self.log(f"Connected to SQL Server: {version}")
            except Exception as e:
                self.log(f"Database connection failed: {str(e)}")
                return

            query = """
            WITH CryptoData AS (
                SELECT 
                    m.id, m.name, m.symbol, m.market_cap_rank,
                    COUNT(d.id) as day_count
                FROM coingecko_crypto_master m
                JOIN coingecko_crypto_daily_data d ON m.id = d.crypto_id
                GROUP BY m.id, m.name, m.symbol, m.market_cap_rank
                HAVING COUNT(d.id) >= ?
            )
            SELECT * FROM CryptoData ORDER BY market_cap_rank
            """
            
            df = pd.read_sql(query, self.engine, params=(self.min_days_required,))
            total_coins = len(df)
            total_models = total_coins * 4  # 4 timeframes per coin
            models_completed = 0
            
            self.log(f"Starting training of {total_models} models ({total_coins} coins x 4 timeframes)")
            self.log(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            import gc
            from tensorflow.keras import backend as K
            from contextlib import contextmanager
            import signal
            
            @contextmanager
            def timeout(seconds):
                def handler(signum, frame):
                    raise TimeoutError(f"Training timed out after {seconds} seconds")
                
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(seconds)
                try:
                    yield
                finally:
                    signal.alarm(0)
            
            for idx, row in df.iterrows():
                coin_id, name = row['id'], row['name']
                coin_start_time = datetime.now()
                
                self.log(f"\nProcessing {name} ({idx+1}/{total_coins}, {(idx/total_coins)*100:.1f}% of coins)")
                
                try:
                    data = pd.read_sql(f"""
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
                        WHERE d.crypto_id = '{coin_id}'
                        ORDER BY d.price_date ASC
                    """, self.engine)
                    
                    if data.empty:
                        self.log(f"No data found for {name}")
                        continue
                        
                    data = data.ffill().fillna(0).infer_objects(copy=False)
                    
                    for days in [1, 2, 3, 7]:
                        try:
                            model_name = f"{coin_id}_LSTM_v1_{days}d"
                            models_completed += 1
                            progress = (models_completed / total_models) * 100
                            
                            elapsed_time = (datetime.now() - start_time).total_seconds() / 3600  # hours
                            estimated_total_time = (elapsed_time / progress) * 100 if progress > 0 else 0
                            remaining_time = max(0, estimated_total_time - elapsed_time)
                            
                            self.log(f"Training {model_name} model... ({progress:.1f}% complete, ~{remaining_time:.1f}h remaining)")
                            
                            X, y, scaler = self.prepare_data(data, days)
                            
                            if len(X) < 50:
                                self.log(f"Insufficient sequences for {model_name}")
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
                                self.log(f"Training timed out for {model_name}")
                                continue
                            
                            model_path = os.path.join(self.models_dir, f"{model_name}.keras")
                            model.save(model_path)
                            
                            scaler_path = os.path.join(self.models_dir, f"{model_name}_scaler.pkl")
                            pd.to_pickle(scaler, scaler_path)
                            
                            # Get the validation metrics
                            val_loss = float(history.history['val_loss'][-1])
                            
                            # Create metric dictionaries
                            mae_metrics = {
                                '24h': val_loss if days == 1 else None,
                                '48h': val_loss if days == 2 else None,
                                '3d': val_loss if days == 3 else None,
                                '7d': val_loss if days == 7 else None
                            }
                            
                            rmse_metrics = mae_metrics.copy()  # Using same values for now
                            
                            # Save performance metrics
                            self.save_model_performance(
                                model_name, 
                                mae_metrics, 
                                rmse_metrics, 
                                len(X), 
                                f"Model trained for {coin_id} with {days}d prediction window"
                            )
                            
                            self.log(f"Saved {model_name} model and scaler (validation loss: {val_loss:.6f})")
                            
                            del model, history
                            gc.collect()
                            
                        except Exception as e:
                            self.log(f"Error training {model_name}: {str(e)}")
                            continue
                            
                    coin_duration = (datetime.now() - coin_start_time).total_seconds() / 60
                    self.log(f"Completed {name} in {coin_duration:.1f} minutes")
                    
                except Exception as e:
                    self.log(f"Error processing {name}: {str(e)}")
                    continue
                    
            total_duration = (datetime.now() - start_time).total_seconds() / 3600
            self.log(f"\nTraining completed in {total_duration:.1f} hours")
            self.log(f"Successfully trained {models_completed} out of {total_models} models")
            
        except Exception as e:
            self.log(f"Error in train_models: {str(e)}")
        finally:
            try:
                self.engine.dispose()
            except:
                pass
            gc.collect()  # Moved inside finally block

    def save_model_performance(self, model_version, mae_metrics, rmse_metrics, samples=0, notes=""):
        """Save model performance metrics to database"""
        try:
            # Replace NaN/None with 0 for numeric fields
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
            self.log(f"Error saving model performance: {str(e)}")

if __name__ == "__main__":
    try:
        trainer = CryptoModelTrainer()
        trainer.train_models()
    except Exception as e:
        print(f"Fatal error: {str(e)}") 