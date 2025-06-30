import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class HVACDataGenerator:
    """HVAC系统数据生成器"""
    
    def generate_current_data(self):
        now = datetime.now()
        return {
            'timestamp': now,
            'temperature': np.random.uniform(18, 28),
            'humidity': np.random.uniform(40, 70),
            'load': np.random.uniform(30, 90),
            'energy_consumption': np.random.uniform(50, 150),
            'cop': np.random.uniform(2.5, 4.5),
            'status': np.random.choice(['正常', '预警', '故障']),
            'runtime': np.random.randint(100, 10000),
            'maintenance_status': np.random.choice(['正常', '需维护']),
            'ambient_temp': np.random.uniform(10, 35),
            'air_quality': np.random.choice(['优', '良', '一般'])
        }
    
    def generate_historical_data(self, hours=24):
        now = datetime.now()
        timestamps = [now - timedelta(hours=i) for i in range(hours)][::-1]
        data = {
            'timestamp': timestamps,
            'temperature': np.random.uniform(18, 28, hours),
            'humidity': np.random.uniform(40, 70, hours),
            'load': np.random.uniform(30, 90, hours),
            'energy_consumption': np.random.uniform(50, 150, hours),
            'cop': np.random.uniform(2.5, 4.5, hours)
        }
        return pd.DataFrame(data)
    
    def generate_training_data(self, days=30):
        """生成训练数据"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        timestamps = pd.date_range(start=start_time, end=end_time, freq='H')
        
        # 生成更复杂的时间模式
        hour_of_day = np.array([t.hour for t in timestamps])
        day_of_week = np.array([t.weekday() for t in timestamps])
        
        # 工作日/周末模式
        workday_factor = np.where(day_of_week < 5, 1.2, 0.8)
        
        # 日间模式
        day_pattern = 1.0 + 0.4 * np.sin(2 * np.pi * hour_of_day / 24)
        
        # 基础负荷
        base_load = 50.0 * workday_factor * day_pattern
        
        # 添加趋势和季节性
        trend = np.linspace(0, 5, len(timestamps))
        seasonal = 10 * np.sin(2 * np.pi * np.arange(len(timestamps)) / (24 * 7))
        
        # 最终负荷
        load = np.clip(base_load + trend + seasonal + np.random.normal(0, 5, len(timestamps)), 0, 100)
        
        # 相关变量
        temperature = 20 + 8 * np.sin(2 * np.pi * hour_of_day / 24) + np.random.normal(0, 2, len(timestamps))
        humidity = 50 + 20 * np.sin(2 * np.pi * hour_of_day / 24) + np.random.normal(0, 5, len(timestamps))
        energy = load * 2.5 + np.random.normal(0, 10, len(timestamps))
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'load': load,
            'temperature': temperature,
            'humidity': humidity,
            'energy_consumption': energy
        })
    
    def generate_detection_data(self):
        """生成故障检测数据"""
        timestamps = pd.date_range(start=datetime.now() - timedelta(hours=48), 
                                 end=datetime.now(), freq='10min')
        
        # 正常数据
        normal_temp = 22 + 2 * np.sin(2 * np.pi * np.arange(len(timestamps)) / (6 * 24))
        normal_pressure = 1.2 + 0.1 * np.sin(2 * np.pi * np.arange(len(timestamps)) / (6 * 24))
        
        # 添加异常
        anomaly_indices = random.sample(range(len(timestamps)), 5)
        
        temperature = normal_temp.copy()
        pressure = normal_pressure.copy()
        
        for idx in anomaly_indices:
            if random.random() < 0.5:
                # 温度异常
                temperature[idx:idx+6] += np.random.normal(5, 1)
            else:
                # 压力异常
                pressure[idx:idx+6] += np.random.normal(0.3, 0.05)
        
        # 添加噪声
        temperature += np.random.normal(0, 0.5, len(timestamps))
        pressure += np.random.normal(0, 0.02, len(timestamps))
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'temperature': temperature,
            'pressure': pressure
        }) 