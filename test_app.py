import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 测试导入自定义模块
try:
    from algorithms.load_forecasting import LoadForecaster
    from algorithms.energy_optimization import EnergyOptimizer
    from algorithms.fault_detection import FaultDetector
    from algorithms.temperature_control import TemperatureController
    from utils.data_generator import HVACDataGenerator
    from utils.visualization import VisualizationHelper
    print("所有模块导入成功！")
except Exception as e:
    print(f"导入错误: {e}")

# 设置页面配置
st.set_page_config(
    page_title="测试页面",
    page_icon="❄️",
    layout="wide"
)

st.title("测试页面")
st.write("如果看到这个页面，说明应用可以正常运行！")

# 测试创建对象
try:
    data_generator = HVACDataGenerator()
    load_forecaster = LoadForecaster()
    energy_optimizer = EnergyOptimizer()
    fault_detector = FaultDetector()
    temp_controller = TemperatureController()
    viz_helper = VisualizationHelper()
    st.success("所有对象创建成功！")
except Exception as e:
    st.error(f"对象创建错误: {e}")

# 测试数据生成
try:
    current_data = data_generator.generate_current_data()
    st.write("当前数据:", current_data)
except Exception as e:
    st.error(f"数据生成错误: {e}") 