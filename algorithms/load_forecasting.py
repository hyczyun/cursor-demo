import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA

class LoadForecaster:
    """负荷预测算法类"""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=10, random_state=42)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict_single(self, X):
        return self.model.predict(X)

    def prepare_features(self, data):
        """准备特征数据"""
        df = data.copy()
        
        # 时间特征
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['weekday'] = df['timestamp'].dt.weekday
        df['month'] = df['timestamp'].dt.month
        
        # 周期性特征
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        
        # 滞后特征
        df['load_lag1'] = df['load'].shift(1)
        df['load_lag24'] = df['load'].shift(24)
        df['temp_lag1'] = df['temperature'].shift(1)
        
        # 移动平均
        df['load_ma6'] = df['load'].rolling(window=6).mean()
        df['load_ma24'] = df['load'].rolling(window=24).mean()
        
        # 删除NaN值
        df = df.dropna()
        
        # 特征列
        feature_columns = ['hour', 'day', 'weekday', 'month', 'hour_sin', 'hour_cos',
                          'day_sin', 'day_cos', 'temperature', 'humidity',
                          'load_lag1', 'load_lag24', 'temp_lag1', 'load_ma6', 'load_ma24']
        
        return df, feature_columns
    
    def train_random_forest(self, data):
        """训练随机森林模型"""
        df, feature_columns = self.prepare_features(data)
        
        X = df[feature_columns]
        y = df['load']
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scaler = scaler
        
        # 训练模型
        self.model.fit(X_scaled, y)
        
        return self.model
    
    def train_xgboost(self, data):
        """训练XGBoost模型"""
        df, feature_columns = self.prepare_features(data)
        
        X = df[feature_columns]
        y = df['load']
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scaler = scaler
        
        # 训练模型
        model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        
        self.model = model
        
        return self.model
    
    def train_lstm_simulation(self, data):
        """模拟LSTM训练（简化版）"""
        # 这里简化LSTM实现，实际应用中可以使用TensorFlow/Keras
        df, feature_columns = self.prepare_features(data)
        
        X = df[feature_columns]
        y = df['load']
        
        # 使用简单的神经网络模拟
        from sklearn.neural_network import MLPRegressor
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scaler = scaler
        
        model = MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
        model.fit(X_scaled, y)
        
        self.model = model
        
        return self.model
    
    def train_arima(self, data):
        """训练ARIMA模型"""
        # 使用负荷数据训练ARIMA
        load_data = data['load'].dropna()
        
        # 简化的ARIMA模型
        model = ARIMA(load_data, order=(1, 1, 1))
        fitted_model = model.fit()
        
        self.model = fitted_model
        
        return self.model
    
    def predict(self, algorithm, data, hours=24):
        """进行预测"""
        import numpy as np  # 确保 np 可用
        # 直接根据 algorithm 判断并训练模型
        if algorithm == "LSTM神经网络":
            self.train_lstm_simulation(data)
        elif algorithm == "XGBoost":
            self.train_xgboost(data)
        elif algorithm == "随机森林":
            self.train_random_forest(data)
        elif algorithm == "时间序列ARIMA":
            self.train_arima(data)
        
        # 生成预测时间
        last_timestamp = data['timestamp'].iloc[-1]
        future_timestamps = pd.date_range(
            start=last_timestamp + timedelta(hours=1),
            periods=hours,
            freq='H'
        )
        
        if algorithm == "时间序列ARIMA":
            # ARIMA预测
            forecast = self.model.forecast(steps=hours)
            predicted_load = np.clip(forecast, 0, 100)
        else:
            # 其他模型预测
            model = self.model
            
            # 准备预测数据
            df, feature_columns = self.prepare_features(data)
            last_features = df[feature_columns].iloc[-1:]
            
            predicted_load = []
            for i in range(hours):
                import numpy as np  # 再次确保 np 可用
                # 更新特征
                future_hour = (last_timestamp + timedelta(hours=i+1)).hour
                future_day = (last_timestamp + timedelta(hours=i+1)).day
                future_weekday = (last_timestamp + timedelta(hours=i+1)).weekday()
                
                # 创建未来特征
                future_features = last_features.copy()
                future_features['hour'] = future_hour
                future_features['day'] = future_day
                future_features['weekday'] = future_weekday
                future_features['hour_sin'] = np.sin(2 * np.pi * future_hour / 24)
                future_features['hour_cos'] = np.cos(2 * np.pi * future_hour / 24)
                
                # 预测
                X_scaled = self.scaler.transform(future_features)
                pred = model.predict(X_scaled)[0]
                predicted_load.append(max(0, min(100, pred)))
                
                # 更新滞后特征
                if i == 0:
                    last_features['load_lag1'] = data['load'].iloc[-1]
                else:
                    last_features['load_lag1'] = predicted_load[i-1]
        
        return pd.DataFrame({
            'timestamp': future_timestamps,
            'predicted_load': predicted_load
        })
    
    def evaluate_model(self, algorithm, data):
        """评估模型性能"""
        # 分割数据
        split_point = int(len(data) * 0.8)
        train_data = data.iloc[:split_point]
        test_data = data.iloc[split_point:]
        
        # 训练模型
        if algorithm == "LSTM神经网络":
            self.train_lstm_simulation(train_data)
        elif algorithm == "XGBoost":
            self.train_xgboost(train_data)
        elif algorithm == "随机森林":
            self.train_random_forest(train_data)
        elif algorithm == "时间序列ARIMA":
            self.train_arima(train_data)
        
        # 预测
        predictions = self.predict(algorithm, train_data, hours=len(test_data))
        
        # 计算指标
        actual = test_data['load'].values
        predicted = predictions['predicted_load'].values
        
        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        r2 = 1 - np.sum((actual - predicted) ** 2) / np.sum((actual - np.mean(actual)) ** 2)
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R²': r2
        } 