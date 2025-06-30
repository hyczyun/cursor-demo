import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

class VisualizationHelper:
    """可视化工具类"""
    
    def __init__(self):
        self.color_scheme = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd',
            'light': '#8c564b',
            'dark': '#e377c2'
        }
    
    def create_system_dashboard(self, current_data, historical_data):
        """创建系统仪表板"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('实时温度', '系统负荷', '能耗趋势', 'COP变化'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 温度图表
        fig.add_trace(
            go.Scatter(
                x=historical_data['timestamp'],
                y=historical_data['temperature'],
                mode='lines',
                name='温度',
                line=dict(color=self.color_scheme['primary'])
            ),
            row=1, col=1
        )
        
        # 负荷图表
        fig.add_trace(
            go.Scatter(
                x=historical_data['timestamp'],
                y=historical_data['load'],
                mode='lines',
                name='负荷',
                line=dict(color=self.color_scheme['secondary'])
            ),
            row=1, col=2
        )
        
        # 能耗图表
        fig.add_trace(
            go.Scatter(
                x=historical_data['timestamp'],
                y=historical_data['energy_consumption'],
                mode='lines',
                name='能耗',
                line=dict(color=self.color_scheme['warning'])
            ),
            row=2, col=1
        )
        
        # COP图表
        fig.add_trace(
            go.Scatter(
                x=historical_data['timestamp'],
                y=historical_data['cop'],
                mode='lines',
                name='COP',
                line=dict(color=self.color_scheme['success'])
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            title_text="HVAC系统实时监控仪表板"
        )
        
        return fig
    
    def create_forecast_comparison(self, historical_data, predictions, algorithm_name):
        """创建预测对比图"""
        fig = go.Figure()
        
        # 历史数据
        fig.add_trace(go.Scatter(
            x=historical_data['timestamp'][-48:],
            y=historical_data['load'][-48:],
            mode='lines',
            name='历史负荷',
            line=dict(color=self.color_scheme['primary'], width=2)
        ))
        
        # 预测数据
        fig.add_trace(go.Scatter(
            x=predictions['timestamp'],
            y=predictions['predicted_load'],
            mode='lines',
            name=f'{algorithm_name}预测',
            line=dict(color=self.color_scheme['warning'], width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=f"{algorithm_name}负荷预测结果",
            xaxis_title="时间",
            yaxis_title="负荷 (%)",
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    def create_optimization_results(self, optimization_results):
        """创建优化结果图"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('优化前后对比', '能耗趋势'),
            specs=[[{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # 优化前后对比
        fig.add_trace(
            go.Bar(
                x=['优化前', '优化后'],
                y=[optimization_results['before_energy'], optimization_results['after_energy']],
                name='能耗 (kWh)',
                marker_color=[self.color_scheme['warning'], self.color_scheme['success']]
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
                line=dict(color=self.color_scheme['primary'])
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            title_text="能耗优化结果"
        )
        
        return fig
    
    def create_fault_detection_plot(self, detection_results):
        """创建故障检测图"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('传感器数据', '异常检测结果'),
            vertical_spacing=0.1
        )
        
        # 传感器数据
        fig.add_trace(
            go.Scatter(
                x=detection_results['timestamp'],
                y=detection_results['temperature'],
                mode='lines',
                name='温度',
                line=dict(color=self.color_scheme['primary'])
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=detection_results['timestamp'],
                y=detection_results['pressure'],
                mode='lines',
                name='压力',
                line=dict(color=self.color_scheme['secondary']),
                yaxis='y2'
            ),
            row=1, col=1
        )
        
        # 异常检测结果
        colors = ['green' if not anomaly else 'red' 
                 for anomaly in detection_results['anomalies']]
        
        fig.add_trace(
            go.Scatter(
                x=detection_results['timestamp'],
                y=detection_results['scores'],
                mode='markers',
                name='异常分数',
                marker=dict(color=colors, size=8)
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            title_text="故障检测结果"
        )
        
        return fig
    
    def create_control_simulation(self, control_results):
        """创建控制仿真图"""
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
                line=dict(color=self.color_scheme['primary'])
            ),
            row=1, col=1
        )
        
        fig.add_hline(
            y=control_results['target'][0],
            line_dash="dash",
            line_color=self.color_scheme['warning'],
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
                line=dict(color=self.color_scheme['success'])
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=500,
            title_text="温度控制仿真结果"
        )
        
        return fig
    
    def create_performance_metrics(self, metrics_data):
        """创建性能指标图"""
        fig = go.Figure()
        
        # 创建雷达图
        categories = list(metrics_data.keys())
        values = list(metrics_data.values())
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='性能指标',
            line_color=self.color_scheme['primary']
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(values) * 1.2]
                )),
            showlegend=False,
            title_text="算法性能指标"
        )
        
        return fig
    
    def create_algorithm_comparison(self, comparison_data):
        """创建算法对比图"""
        algorithms = list(comparison_data.keys())
        metrics = ['MAE', 'RMSE', 'MAPE', 'R²']
        
        fig = go.Figure()
        
        for i, metric in enumerate(metrics):
            values = [comparison_data[algo][metric] for algo in algorithms]
            
            fig.add_trace(go.Bar(
                name=metric,
                x=algorithms,
                y=values,
                marker_color=self.color_scheme[list(self.color_scheme.keys())[i]]
            ))
        
        fig.update_layout(
            title_text="算法性能对比",
            barmode='group',
            height=400
        )
        
        return fig
    
    def create_energy_flow_diagram(self, energy_data):
        """创建能量流图"""
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=["输入电能", "压缩机", "冷凝器", "蒸发器", "冷却水", "室内空气", "热损失"],
                color="blue"
            ),
            link=dict(
                source=[0, 1, 1, 2, 3, 3],
                target=[1, 2, 3, 4, 5, 6],
                value=[energy_data['input_power'], 
                       energy_data['compressor_loss'],
                       energy_data['refrigeration_effect'],
                       energy_data['condenser_heat'],
                       energy_data['cooling_effect'],
                       energy_data['heat_loss']]
            ))])
        
        fig.update_layout(
            title_text="HVAC系统能量流图",
            font_size=10,
            height=500
        )
        
        return fig
    
    def create_3d_surface_plot(self, x_data, y_data, z_data, title="3D表面图"):
        """创建3D表面图"""
        fig = go.Figure(data=[go.Surface(z=z_data, x=x_data, y=y_data)])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X轴',
                yaxis_title='Y轴',
                zaxis_title='Z轴'
            ),
            height=500
        )
        
        return fig
    
    def create_heatmap(self, data, x_labels, y_labels, title="热力图"):
        """创建热力图"""
        fig = go.Figure(data=go.Heatmap(
            z=data,
            x=x_labels,
            y=y_labels,
            colorscale='Viridis'
        ))
        
        fig.update_layout(
            title=title,
            height=400
        )
        
        return fig
    
    def create_gauge_chart(self, value, min_val, max_val, title="仪表盘"):
        """创建仪表盘"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title},
            delta={'reference': (min_val + max_val) / 2},
            gauge={
                'axis': {'range': [min_val, max_val]},
                'bar': {'color': self.color_scheme['primary']},
                'steps': [
                    {'range': [min_val, (min_val + max_val) * 0.3], 'color': "lightgray"},
                    {'range': [(min_val + max_val) * 0.3, (min_val + max_val) * 0.7], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': max_val * 0.9
                }
            }
        ))
        
        fig.update_layout(height=300)
        
        return fig
    
    def create_timeline_chart(self, events_data):
        """创建时间线图"""
        fig = go.Figure()
        
        for event in events_data:
            fig.add_trace(go.Scatter(
                x=[event['time'], event['time']],
                y=[0, 1],
                mode='markers+text',
                name=event['name'],
                text=[event['name']],
                textposition="top center",
                marker=dict(size=10, color=event['color'])
            ))
        
        fig.update_layout(
            title="系统事件时间线",
            xaxis_title="时间",
            yaxis=dict(showticklabels=False),
            height=300
        )
        
        return fig
    
    def plot_timeseries(self, df, y, title):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df[y], mode='lines', name=y))
        fig.update_layout(title=title, xaxis_title='时间', yaxis_title=y)
        return fig 