import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM
import xgboost as xgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class FaultDetector:
    """故障检测算法类"""
    
    def __init__(self):
        self.model = IsolationForest(contamination=0.1, random_state=42)
        self.scalers = {}
        self.thresholds = {}
        self.fault_history = []
        
    def prepare_features(self, data):
        """准备故障检测特征"""
        df = data.copy()
        
        # 基础特征
        features = ['temperature', 'humidity', 'load', 'energy_consumption', 'cop']
        
        # 统计特征
        for col in features:
            if col in df.columns:
                # 移动统计量
                df[f'{col}_mean_6h'] = df[col].rolling(window=6).mean()
                df[f'{col}_std_6h'] = df[col].rolling(window=6).std()
                df[f'{col}_min_6h'] = df[col].rolling(window=6).min()
                df[f'{col}_max_6h'] = df[col].rolling(window=6).max()
                
                # 变化率
                df[f'{col}_diff'] = df[col].diff()
                df[f'{col}_diff_abs'] = df[col].diff().abs()
                
                # 异常分数
                df[f'{col}_zscore'] = np.abs(stats.zscore(df[col].fillna(method='ffill')))
        
        # 系统效率指标
        if 'energy_consumption' in df.columns and 'load' in df.columns:
            df['efficiency_ratio'] = df['load'] / (df['energy_consumption'] + 1e-6)
            df['efficiency_ratio_mean'] = df['efficiency_ratio'].rolling(window=6).mean()
            df['efficiency_ratio_std'] = df['efficiency_ratio'].rolling(window=6).std()
        
        # 删除NaN值
        df = df.dropna()
        
        # 选择特征列
        feature_columns = [col for col in df.columns if any(
            keyword in col for keyword in ['_mean_', '_std_', '_min_', '_max_', 
                                         '_diff', '_zscore', '_ratio']
        )]
        
        # 确保基础特征也在其中
        for col in features:
            if col in df.columns and col not in feature_columns:
                feature_columns.append(col)
        
        return df, feature_columns
    
    def train(self, X):
        self.model.fit(X)

    def detect(self, X):
        return self.model.predict(X)
    
    def train_one_class_svm(self, data):
        """训练单类SVM模型"""
        df, feature_columns = self.prepare_features(data)
        
        X = df[feature_columns]
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 训练模型
        model = OneClassSVM(
            kernel='rbf',
            nu=0.1,  # 异常比例
            gamma='scale'
        )
        model.fit(X_scaled)
        
        self.scalers['one_class_svm'] = scaler
        
        return model
    
    def train_rule_based(self, data):
        """基于规则的故障检测"""
        # 设置阈值
        thresholds = {
            'temperature_high': 30,
            'temperature_low': 15,
            'humidity_high': 80,
            'humidity_low': 30,
            'load_high': 95,
            'load_low': 10,
            'cop_low': 2.0,
            'efficiency_ratio_low': 0.3
        }
        
        self.thresholds['rule_based'] = thresholds
        return thresholds
    
    def detect_faults(self, algorithm, data):
        """检测故障"""
        if algorithm not in self.scalers:
            # 训练模型
            if algorithm == "隔离森林":
                self.train(data)
            elif algorithm == "单类SVM":
                self.train_one_class_svm(data)
            elif algorithm == "基于规则":
                self.train_rule_based(data)
        
        df, feature_columns = self.prepare_features(data)
        
        if algorithm == "基于规则":
            return self._rule_based_detection(df)
        else:
            return self._anomaly_detection_detection(df, feature_columns, algorithm)
    
    def _rule_based_detection(self, df):
        """基于规则的故障检测"""
        thresholds = self.thresholds['rule_based']
        faults = []
        
        for idx, row in df.iterrows():
            fault_flags = []
            
            # 温度异常
            if row['temperature'] > thresholds['temperature_high']:
                fault_flags.append(f"温度过高: {row['temperature']:.1f}°C")
            elif row['temperature'] < thresholds['temperature_low']:
                fault_flags.append(f"温度过低: {row['temperature']:.1f}°C")
            
            # 湿度异常
            if row['humidity'] > thresholds['humidity_high']:
                fault_flags.append(f"湿度过高: {row['humidity']:.1f}%")
            elif row['humidity'] < thresholds['humidity_low']:
                fault_flags.append(f"湿度过低: {row['humidity']:.1f}%")
            
            # 负荷异常
            if row['load'] > thresholds['load_high']:
                fault_flags.append(f"负荷过高: {row['load']:.1f}%")
            elif row['load'] < thresholds['load_low']:
                fault_flags.append(f"负荷过低: {row['load']:.1f}%")
            
            # COP异常
            if row['cop'] < thresholds['cop_low']:
                fault_flags.append(f"COP过低: {row['cop']:.2f}")
            
            # 效率异常
            if 'efficiency_ratio' in row and row['efficiency_ratio'] < thresholds['efficiency_ratio_low']:
                fault_flags.append(f"效率过低: {row['efficiency_ratio']:.2f}")
            
            if fault_flags:
                faults.append({
                    'timestamp': row['timestamp'],
                    'fault_type': '规则检测',
                    'severity': 'high' if len(fault_flags) > 2 else 'medium',
                    'description': '; '.join(fault_flags),
                    'confidence': 0.8
                })
        
        return faults
    
    def _anomaly_detection_detection(self, df, feature_columns, algorithm):
        """异常检测算法检测"""
        scaler = self.scalers[algorithm]
        
        X = df[feature_columns]
        X_scaled = scaler.transform(X)
        
        if algorithm == "隔离森林":
            # 隔离森林返回-1表示异常，1表示正常
            predictions = self.model.predict(X_scaled)
            anomaly_scores = self.model.decision_function(X_scaled)
        elif algorithm == "单类SVM":
            # SVM返回-1表示异常，1表示正常
            predictions = self.scalers[algorithm].predict(X_scaled)
            anomaly_scores = self.scalers[algorithm].decision_function(X_scaled)
        
        faults = []
        for idx, (pred, score) in enumerate(zip(predictions, anomaly_scores)):
            if pred == -1:  # 异常
                # 将异常分数转换为置信度
                confidence = min(1.0, max(0.0, abs(score) / 2))
                severity = 'high' if confidence > 0.8 else 'medium' if confidence > 0.6 else 'low'
                
                faults.append({
                    'timestamp': df.iloc[idx]['timestamp'],
                    'fault_type': f'{algorithm}检测',
                    'severity': severity,
                    'description': f"异常检测算法识别到异常，异常分数: {score:.3f}",
                    'confidence': confidence
                })
        
        return faults
    
    def get_fault_statistics(self, faults):
        """获取故障统计信息"""
        if not faults:
            return {
                'total_faults': 0,
                'high_severity': 0,
                'medium_severity': 0,
                'low_severity': 0,
                'fault_types': {},
                'avg_confidence': 0
            }
        
        stats = {
            'total_faults': len(faults),
            'high_severity': len([f for f in faults if f['severity'] == 'high']),
            'medium_severity': len([f for f in faults if f['severity'] == 'medium']),
            'low_severity': len([f for f in faults if f['severity'] == 'low']),
            'fault_types': {},
            'avg_confidence': np.mean([f['confidence'] for f in faults])
        }
        
        # 统计故障类型
        for fault in faults:
            fault_type = fault['fault_type']
            if fault_type not in stats['fault_types']:
                stats['fault_types'][fault_type] = 0
            stats['fault_types'][fault_type] += 1
        
        return stats
    
    def get_recommendations(self, faults):
        """根据故障生成建议"""
        recommendations = []
        
        for fault in faults:
            if '温度过高' in fault['description']:
                recommendations.append({
                    'priority': 'high' if fault['severity'] == 'high' else 'medium',
                    'action': '检查制冷系统，清洁冷凝器，检查制冷剂',
                    'fault_type': fault['fault_type']
                })
            elif '温度过低' in fault['description']:
                recommendations.append({
                    'priority': 'high' if fault['severity'] == 'high' else 'medium',
                    'action': '检查加热系统，调整温度设定值',
                    'fault_type': fault['fault_type']
                })
            elif 'COP过低' in fault['description']:
                recommendations.append({
                    'priority': 'high',
                    'action': '检查压缩机效率，清洁换热器，检查制冷剂泄漏',
                    'fault_type': fault['fault_type']
                })
            elif '负荷过高' in fault['description']:
                recommendations.append({
                    'priority': 'medium',
                    'action': '检查系统容量，考虑增加设备或优化负荷分配',
                    'fault_type': fault['fault_type']
                })
            else:
                recommendations.append({
                    'priority': 'medium',
                    'action': '进行系统全面检查，查看详细日志',
                    'fault_type': fault['fault_type']
                })
        
        return recommendations
