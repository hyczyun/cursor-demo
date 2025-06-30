import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import time

# 导入自定义模块
from algorithms.load_forecasting import LoadForecaster
from algorithms.energy_optimization import EnergyOptimizer
from algorithms.fault_detection import FaultDetector
from algorithms.temperature_control import TemperatureController
from utils.data_generator import HVACDataGenerator
from utils.visualization import VisualizationHelper

# 设置页面配置
st.set_page_config(
    page_title="工业暖通制冷站AI算法演示平台",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .algorithm-section {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class HVACDemoPlatform:
    def __init__(self):
        self.data_generator = HVACDataGenerator()
        self.load_forecaster = LoadForecaster()
        self.energy_optimizer = EnergyOptimizer()
        self.fault_detector = FaultDetector()
        self.temp_controller = TemperatureController()
        self.viz_helper = VisualizationHelper()
        
    def main(self):
        # 主标题
        st.markdown('<h1 class="main-header">❄️ 工业暖通制冷站AI算法演示平台</h1>', unsafe_allow_html=True)
        
        # 侧边栏
        self.sidebar()
        
        # 主界面
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 系统概览", 
            "🔮 负荷预测", 
            "⚡ 能耗优化", 
            "🔍 故障诊断", 
            "🌡️ 温度控制"
        ])
        
        with tab1:
            self.system_overview()
        
        with tab2:
            self.load_forecasting_demo()
            
        with tab3:
            self.energy_optimization_demo()
            
        with tab4:
            self.fault_detection_demo()
            
        with tab5:
            self.temperature_control_demo()
    
    def sidebar(self):
        st.sidebar.title("⚙️ 系统设置")
        
        # 系统参数设置
        st.sidebar.subheader("系统参数")
        self.cooling_capacity = st.sidebar.slider(
            "制冷量 (kW)", 
            min_value=100, 
            max_value=1000, 
            value=500, 
            step=50
        )
        
        self.target_temp = st.sidebar.slider(
            "目标温度 (°C)", 
            min_value=18, 
            max_value=26, 
            value=22, 
            step=1
        )
        
        self.humidity = st.sidebar.slider(
            "相对湿度 (%)", 
            min_value=30, 
            max_value=80, 
            value=50, 
            step=5
        )
        
        # 算法参数
        st.sidebar.subheader("算法参数")
        self.prediction_horizon = st.sidebar.selectbox(
            "预测时间范围",
            ["1小时", "6小时", "24小时", "7天"]
        )
        
        self.optimization_goal = st.sidebar.selectbox(
            "优化目标",
            ["能耗最小化", "舒适度最大化", "平衡优化"]
        )
        
        # 实时监控开关
        st.sidebar.subheader("实时监控")
        self.real_time_monitoring = st.sidebar.checkbox("启用实时监控", value=True)
        
        if self.real_time_monitoring:
            st.sidebar.success("✅ 实时监控已启用")
        else:
            st.sidebar.warning("⚠️ 实时监控已禁用")
    
    def system_overview(self):
        st.header("📊 系统概览")
        
        # 生成实时数据
        current_data = self.data_generator.generate_current_data()
        
        # 关键指标卡片
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>当前温度</h3>
                <h2>{current_data['temperature']:.1f}°C</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>系统负荷</h3>
                <h2>{current_data['load']:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>能耗</h3>
                <h2>{current_data['energy_consumption']:.1f} kWh</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>系统状态</h3>
                <h2>{current_data['status']}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # 实时数据图表
        st.subheader("实时运行数据")
        
        # 生成历史数据
        historical_data = self.data_generator.generate_historical_data(hours=24)
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('温度变化', '负荷变化', '能耗变化', 'COP变化'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 温度图表
        fig.add_trace(
            go.Scatter(x=historical_data['timestamp'], y=historical_data['temperature'],
                      mode='lines', name='实际温度', line=dict(color='red')),
            row=1, col=1
        )
        fig.add_hline(y=self.target_temp, line_dash="dash", line_color="blue",
                     annotation_text="目标温度", row=1, col=1)
        
        # 负荷图表
        fig.add_trace(
            go.Scatter(x=historical_data['timestamp'], y=historical_data['load'],
                      mode='lines', name='系统负荷', line=dict(color='green')),
            row=1, col=2
        )
        
        # 能耗图表
        fig.add_trace(
            go.Scatter(x=historical_data['timestamp'], y=historical_data['energy_consumption'],
                      mode='lines', name='能耗', line=dict(color='orange')),
            row=2, col=1
        )
        
        # COP图表
        fig.add_trace(
            go.Scatter(x=historical_data['timestamp'], y=historical_data['cop'],
                      mode='lines', name='COP', line=dict(color='purple')),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # 系统状态详情
        st.subheader("系统状态详情")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**设备运行状态**")
            status_data = {
                "压缩机": "正常运行",
                "冷凝器": "正常运行", 
                "蒸发器": "正常运行",
                "冷却塔": "正常运行",
                "水泵": "正常运行"
            }
            
            for device, status in status_data.items():
                if status == "正常运行":
                    st.success(f"✅ {device}: {status}")
                else:
                    st.error(f"❌ {device}: {status}")
        
        with col2:
            st.write("**关键参数**")
            param_data = {
                "压缩机频率": f"{current_data['compressor_frequency']:.1f} Hz",
                "冷却水流量": f"{current_data['water_flow']:.1f} m³/h",
                "冷凝压力": f"{current_data['condenser_pressure']:.1f} MPa",
                "蒸发压力": f"{current_data['evaporator_pressure']:.1f} MPa"
            }
            
            for param, value in param_data.items():
                st.info(f"📊 {param}: {value}")
    
    def load_forecasting_demo(self):
        st.header("🔮 负荷预测算法演示")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("负荷预测模型")
            
            # 选择预测算法
            algorithm = st.selectbox(
                "选择预测算法",
                ["LSTM神经网络", "XGBoost", "随机森林", "时间序列ARIMA"]
            )
            
            # 生成训练数据
            training_data = self.data_generator.generate_training_data(days=30)
            
            # 训练模型
            if st.button("训练预测模型"):
                with st.spinner("正在训练模型..."):
                    # 模拟训练过程
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    # 获取预测结果
                    predictions = self.load_forecaster.predict(
                        algorithm, training_data, hours=24
                    )
                    
                    st.success("模型训练完成！")
                    
                    # 显示预测结果
                    fig = go.Figure()
                    
                    # 历史数据
                    fig.add_trace(go.Scatter(
                        x=training_data['timestamp'][-48:],
                        y=training_data['load'][-48:],
                        mode='lines',
                        name='历史负荷',
                        line=dict(color='blue')
                    ))
                    
                    # 预测数据
                    fig.add_trace(go.Scatter(
                        x=predictions['timestamp'],
                        y=predictions['predicted_load'],
                        mode='lines',
                        name=f'{algorithm}预测',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig.update_layout(
                        title="负荷预测结果",
                        xaxis_title="时间",
                        yaxis_title="负荷 (%)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("预测性能指标")
            
            # 模拟性能指标
            metrics = {
                "MAE": 2.34,
                "RMSE": 3.12,
                "MAPE": 4.56,
                "R²": 0.89
            }
            
            for metric, value in metrics.items():
                st.metric(metric, f"{value:.2f}")
            
            st.subheader("算法说明")
            st.info("""
            **LSTM神经网络**: 适合处理时间序列数据，能够捕捉长期依赖关系
            
            **XGBoost**: 梯度提升算法，处理非线性关系能力强
            
            **随机森林**: 集成学习方法，抗过拟合能力强
            
            **ARIMA**: 经典时间序列模型，适合趋势和季节性预测
            """)
    
    def energy_optimization_demo(self):
        st.header("⚡ 能耗优化算法演示")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("优化策略")
            
            # 优化算法选择
            optimization_algorithm = st.selectbox(
                "选择优化算法",
                ["遗传算法", "粒子群优化", "强化学习", "多目标优化"]
            )
            
            # 优化目标权重
            st.write("**优化目标权重**")
            energy_weight = st.slider("能耗权重", 0.0, 1.0, 0.6, 0.1)
            comfort_weight = st.slider("舒适度权重", 0.0, 1.0, 0.3, 0.1)
            cost_weight = st.slider("成本权重", 0.0, 1.0, 0.1, 0.1)
            
            # 运行优化
            if st.button("运行能耗优化"):
                with st.spinner("正在优化..."):
                    # 模拟优化过程
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.02)
                        progress_bar.progress(i + 1)
                    
                    # 获取优化结果
                    optimization_results = self.energy_optimizer.optimize(
                        algorithm=optimization_algorithm,
                        energy_weight=energy_weight,
                        comfort_weight=comfort_weight,
                        cost_weight=cost_weight
                    )
                    
                    st.success("优化完成！")
                    
                    # 显示优化结果
                    fig = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=('优化前后对比', '能耗趋势'),
                        specs=[[{"type": "bar"}, {"type": "scatter"}]]
                    )
                    
                    # 优化前后对比
                    fig.add_trace(
                        go.Bar(
                            x=['优化前', '优化后'],
                            y=[optimization_results['before_energy'], 
                               optimization_results['after_energy']],
                            name='能耗 (kWh)',
                            marker_color=['red', 'green']
                        ),
                        row=1, col=1
                    )
                    
                    # 能耗趋势
                    fig.add_trace(
                        go.Scatter(
                            x=optimization_results['timeline'],
                            y=optimization_results['energy_trend'],
                            mode='lines',
                            name='能耗趋势',
                            line=dict(color='blue')
                        ),
                        row=1, col=2
                    )
                    
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("优化效果")
            
            # 显示优化指标
            improvement = 15.6
            st.metric("能耗降低", f"{improvement:.1f}%", delta=f"-{improvement:.1f}%")
            
            st.metric("舒适度提升", "8.3%", delta="+8.3%")
            st.metric("成本节约", "12.4%", delta="-12.4%")
            
            st.subheader("优化建议")
            st.info("""
            **当前优化建议**:
            
            1. 调整压缩机运行频率
            2. 优化冷却水流量
            3. 改进启停策略
            4. 调整温度设定点
            """)
    
    def fault_detection_demo(self):
        st.header("🔍 故障诊断算法演示")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("故障检测")
            
            # 检测算法选择
            detection_algorithm = st.selectbox(
                "选择检测算法",
                ["隔离森林", "自编码器", "One-Class SVM", "LSTM异常检测"]
            )
            
            # 生成检测数据
            detection_data = self.data_generator.generate_detection_data()
            
            # 运行故障检测
            if st.button("运行故障检测"):
                with st.spinner("正在检测..."):
                    # 模拟检测过程
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    # 获取检测结果
                    detection_results = self.fault_detector.detect(
                        algorithm=detection_algorithm,
                        data=detection_data
                    )
                    
                    st.success("检测完成！")
                    
                    # 显示检测结果
                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=('传感器数据', '异常检测结果'),
                        vertical_spacing=0.1
                    )
                    
                    # 传感器数据
                    fig.add_trace(
                        go.Scatter(
                            x=detection_data['timestamp'],
                            y=detection_data['temperature'],
                            mode='lines',
                            name='温度',
                            line=dict(color='blue')
                        ),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=detection_data['timestamp'],
                            y=detection_data['pressure'],
                            mode='lines',
                            name='压力',
                            line=dict(color='red'),
                            yaxis='y2'
                        ),
                        row=1, col=1
                    )
                    
                    # 异常检测结果
                    colors = ['green' if not anomaly else 'red' 
                             for anomaly in detection_results['anomalies']]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=detection_data['timestamp'],
                            y=detection_results['scores'],
                            mode='markers',
                            name='异常分数',
                            marker=dict(color=colors, size=8)
                        ),
                        row=2, col=1
                    )
                    
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("检测结果")
            
            # 显示检测统计
            total_points = len(detection_results['anomalies'])
            anomaly_count = sum(detection_results['anomalies'])
            anomaly_rate = (anomaly_count / total_points) * 100
            
            st.metric("检测点数", total_points)
            st.metric("异常点数", anomaly_count)
            st.metric("异常率", f"{anomaly_rate:.2f}%")
            
            # 故障类型分析
            if anomaly_count > 0:
                st.subheader("故障类型分析")
                
                fault_types = {
                    "传感器故障": 0.4,
                    "设备磨损": 0.3,
                    "控制异常": 0.2,
                    "环境因素": 0.1
                }
                
                for fault_type, probability in fault_types.items():
                    st.progress(probability)
                    st.write(f"{fault_type}: {probability*100:.1f}%")
            
            st.subheader("维护建议")
            st.warning("""
            **检测到异常，建议**:
            
            1. 检查传感器连接
            2. 清洁设备部件
            3. 校准控制系统
            4. 联系技术人员
            """)
    
    def temperature_control_demo(self):
        st.header("🌡️ 温度控制算法演示")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("智能温度控制")
            
            # 控制算法选择
            control_algorithm = st.selectbox(
                "选择控制算法",
                ["PID控制", "模糊控制", "神经网络控制", "自适应控制"]
            )
            
            # 控制参数
            st.write("**控制参数**")
            kp = st.slider("比例系数 Kp", 0.1, 10.0, 2.0, 0.1)
            ki = st.slider("积分系数 Ki", 0.01, 1.0, 0.1, 0.01)
            kd = st.slider("微分系数 Kd", 0.01, 1.0, 0.05, 0.01)
            
            # 运行控制仿真
            if st.button("运行控制仿真"):
                with st.spinner("正在仿真..."):
                    # 模拟控制过程
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    # 获取控制结果
                    control_results = self.temp_controller.simulate(
                        algorithm=control_algorithm,
                        target_temp=self.target_temp,
                        kp=kp, ki=ki, kd=kd
                    )
                    
                    st.success("仿真完成！")
                    
                    # 显示控制结果
                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=('温度响应', '控制信号'),
                        vertical_spacing=0.1
                    )
                    
                    # 温度响应
                    fig.add_trace(
                        go.Scatter(
                            x=control_results['time'],
                            y=control_results['temperature'],
                            mode='lines',
                            name='实际温度',
                            line=dict(color='red')
                        ),
                        row=1, col=1
                    )
                    
                    fig.add_hline(
                        y=self.target_temp,
                        line_dash="dash",
                        line_color="blue",
                        annotation_text="目标温度",
                        row=1, col=1
                    )
                    
                    # 控制信号
                    fig.add_trace(
                        go.Scatter(
                            x=control_results['time'],
                            y=control_results['control_signal'],
                            mode='lines',
                            name='控制信号',
                            line=dict(color='green')
                        ),
                        row=2, col=1
                    )
                    
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("控制性能")
            
            # 显示控制指标
            settling_time = 45.2
            overshoot = 2.1
            steady_state_error = 0.3
            
            st.metric("调节时间", f"{settling_time:.1f}秒")
            st.metric("超调量", f"{overshoot:.1f}°C")
            st.metric("稳态误差", f"{steady_state_error:.1f}°C")
            
            # 性能评估
            if overshoot < 3 and steady_state_error < 0.5:
                st.success("✅ 控制性能良好")
            elif overshoot < 5 and steady_state_error < 1.0:
                st.warning("⚠️ 控制性能一般")
            else:
                st.error("❌ 控制性能较差")
            
            st.subheader("算法特点")
            st.info("""
            **PID控制**: 经典控制算法，响应快速
            
            **模糊控制**: 处理非线性，鲁棒性强
            
            **神经网络控制**: 自适应能力强
            
            **自适应控制**: 参数自调整
            """)

if __name__ == "__main__":
    platform = HVACDemoPlatform()
    platform.main() 