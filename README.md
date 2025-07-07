# 工业暖通制冷站AI算法演示平台

一个基于Streamlit的智能暖通空调系统演示平台，集成了多种AI算法用于负荷预测、能耗优化、故障诊断和温度控制。

## 🚀 功能特性

- **📊 系统概览**: 实时监控系统运行状态和关键指标
- **🔮 负荷预测**: 支持LSTM、XGBoost、随机森林、ARIMA等多种算法
- **⚡ 能耗优化**: 遗传算法、粒子群优化、强化学习等优化策略
- **🔍 故障诊断**: 异常检测和故障诊断算法
- **🌡️ 温度控制**: PID、模糊控制、神经网络等智能控制算法

## 🛠️ 技术栈

- **前端**: Streamlit
- **数据处理**: Pandas, NumPy
- **机器学习**: Scikit-learn, XGBoost, LightGBM
- **可视化**: Plotly, Matplotlib, Seaborn
- **时间序列**: Statsmodels

## 📦 安装说明

### 本地运行

1. 克隆仓库：
```bash
git clone <your-repository-url>
cd cursor-demo
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 运行应用：
```bash
streamlit run app.py
```

4. 在浏览器中打开：`http://localhost:8501`

## 🎯 使用方法

1. **系统概览**: 查看实时运行数据和系统状态
2. **负荷预测**: 选择算法训练模型并查看预测结果
3. **能耗优化**: 设置优化目标权重并运行优化算法
4. **故障诊断**: 选择检测算法识别系统异常
5. **温度控制**: 调整控制参数并运行控制仿真

## 📁 项目结构

```
cursor-demo/
├── app.py                 # 主应用文件
├── requirements.txt       # 依赖包列表
├── README.md             # 项目说明
├── algorithms/           # AI算法模块
│   ├── load_forecasting.py
│   ├── energy_optimization.py
│   ├── fault_detection.py
│   └── temperature_control.py
└── utils/               # 工具模块
    ├── data_generator.py
    └── visualization.py
```

## 🔧 配置说明

在侧边栏中可以调整以下参数：
- 系统参数：制冷量、目标温度、相对湿度
- 算法参数：预测时间范围、优化目标
- 实时监控开关

## 📊 演示数据

应用使用模拟数据演示各种算法功能，包括：
- 历史运行数据
- 传感器数据
- 系统状态信息

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 📄 许可证

MIT License

## 📞 联系方式

如有问题，请通过GitHub Issues联系。