import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.optimize import minimize
import random

class TemperatureController:
    """温度控制算法类"""
    
    def __init__(self):
        self.pid_params = {'kp': 2.0, 'ki': 0.1, 'kd': 0.05}
        self.fuzzy_rules = {}
        self.neural_network = None
        self.control_history = []
        
    def pid_control(self, error, dt=1.0):
        """PID控制器"""
        # 积分项（需要历史数据）
        if not hasattr(self, 'integral'):
            self.integral = 0
        if not hasattr(self, 'prev_error'):
            self.prev_error = 0
        
        # PID计算
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        
        # 积分限幅
        self.integral = np.clip(self.integral, -100, 100)
        
        # 控制输出
        output = (self.pid_params['kp'] * error + 
                 self.pid_params['ki'] * self.integral + 
                 self.pid_params['kd'] * derivative)
        
        self.prev_error = error
        
        return output
    
    def fuzzy_control(self, error, error_rate):
        """模糊控制"""
        # 简化的模糊控制实现
        # 实际应用中可以使用更复杂的模糊逻辑
        
        # 模糊化
        if abs(error) < 0.5:
            error_fuzzy = "小"
        elif abs(error) < 2.0:
            error_fuzzy = "中"
        else:
            error_fuzzy = "大"
        
        if abs(error_rate) < 0.1:
            rate_fuzzy = "小"
        elif abs(error_rate) < 0.5:
            rate_fuzzy = "中"
        else:
            rate_fuzzy = "大"
        
        # 模糊规则（简化版）
        rules = {
            ("小", "小"): 0.1,
            ("小", "中"): 0.2,
            ("小", "大"): 0.3,
            ("中", "小"): 0.4,
            ("中", "中"): 0.6,
            ("中", "大"): 0.8,
            ("大", "小"): 0.7,
            ("大", "中"): 0.9,
            ("大", "大"): 1.0
        }
        
        # 去模糊化
        control_output = rules.get((error_fuzzy, rate_fuzzy), 0.5)
        
        # 根据误差符号调整输出
        if error < 0:
            control_output = -control_output
        
        return control_output * 10  # 缩放输出
    
    def neural_network_control(self, state):
        """神经网络控制"""
        # 简化的神经网络控制器
        # 实际应用中可以使用TensorFlow/Keras实现
        
        # 状态向量：[当前温度, 目标温度, 温度变化率, 时间]
        current_temp, target_temp, temp_rate, hour = state
        
        # 简单的神经网络模拟
        # 输入层 -> 隐藏层 -> 输出层
        input_layer = np.array([current_temp, target_temp, temp_rate, hour])
        
        # 权重矩阵（随机初始化）
        if not hasattr(self, 'weights1'):
            self.weights1 = np.random.randn(4, 8) * 0.1
            self.weights2 = np.random.randn(8, 1) * 0.1
        
        # 前向传播
        hidden_layer = np.tanh(np.dot(input_layer, self.weights1))
        output_layer = np.tanh(np.dot(hidden_layer, self.weights2))
        
        # 控制输出
        control_output = output_layer[0] * 10
        
        return control_output
    
    def adaptive_control(self, error, system_gain_estimate):
        """自适应控制"""
        # 简化的自适应控制器
        
        # 自适应增益
        adaptive_gain = 1.0 / (1.0 + abs(error))
        
        # 控制输出
        control_output = adaptive_gain * system_gain_estimate * error
        
        return control_output
    
    def simulate(self, target_temp, steps=50):
        temps = [np.random.uniform(18, 28)]
        for _ in range(steps-1):
            last = temps[-1]
            error = target_temp - last
            temps.append(last + 0.2 * error + np.random.normal(0, 0.2))
        return temps
    
    def calculate_settling_time(self, temperature_history, target_temp, tolerance=0.1):
        """计算调节时间"""
        for i, temp in enumerate(temperature_history):
            if abs(temp - target_temp) <= tolerance:
                # 检查是否在容差范围内持续
                sustained = True
                for j in range(i, min(i + 10, len(temperature_history))):
                    if abs(temperature_history[j] - target_temp) > tolerance:
                        sustained = False
                        break
                if sustained:
                    return i
        return len(temperature_history)
    
    def calculate_overshoot(self, temperature_history, target_temp):
        """计算超调量"""
        if len(temperature_history) < 2:
            return 0
        
        max_temp = max(temperature_history)
        if max_temp > target_temp:
            return max_temp - target_temp
        return 0
    
    def calculate_steady_state_error(self, temperature_history, target_temp):
        """计算稳态误差"""
        # 使用最后20%的数据计算稳态误差
        start_idx = int(len(temperature_history) * 0.8)
        steady_state_temps = temperature_history[start_idx:]
        
        if len(steady_state_temps) == 0:
            return 0
        
        mean_temp = np.mean(steady_state_temps)
        return abs(mean_temp - target_temp)
    
    def optimize_parameters(self, algorithm, target_temp, optimization_method="genetic"):
        """优化控制参数"""
        if algorithm != "PID控制":
            return self.pid_params
        
        def objective_function(params):
            kp, ki, kd = params
            
            # 运行仿真
            simulation = self.simulate(target_temp, duration=50)
            
            # 计算目标函数（最小化调节时间、超调量和稳态误差）
            settling_time = self.calculate_settling_time(simulation, target_temp)
            overshoot = self.calculate_overshoot(simulation, target_temp)
            steady_state_error = self.calculate_steady_state_error(simulation, target_temp)
            
            # 加权目标函数
            objective = (0.4 * settling_time / 50 + 
                        0.4 * overshoot / 5 + 
                        0.2 * steady_state_error / 2)
            
            return objective
        
        # 参数范围
        bounds = [(0.1, 10.0), (0.01, 1.0), (0.01, 1.0)]
        
        if optimization_method == "genetic":
            # 简化的遗传算法优化
            best_params = self.genetic_optimization(objective_function, bounds)
        else:
            # 使用scipy优化
            result = minimize(objective_function, [2.0, 0.1, 0.05], bounds=bounds)
            best_params = result.x
        
        return {'kp': best_params[0], 'ki': best_params[1], 'kd': best_params[2]}
    
    def genetic_optimization(self, objective_function, bounds, population_size=20, generations=50):
        """遗传算法优化参数"""
        # 初始化种群
        population = []
        for _ in range(population_size):
            individual = [random.uniform(bounds[i][0], bounds[i][1]) for i in range(len(bounds))]
            population.append(individual)
        
        best_individual = None
        best_fitness = float('inf')
        
        for generation in range(generations):
            # 计算适应度
            fitness_scores = []
            for individual in population:
                try:
                    fitness = objective_function(individual)
                    fitness_scores.append(fitness)
                except:
                    fitness_scores.append(float('inf'))
            
            # 找到最佳个体
            min_idx = np.argmin(fitness_scores)
            if fitness_scores[min_idx] < best_fitness:
                best_fitness = fitness_scores[min_idx]
                best_individual = population[min_idx].copy()
            
            # 选择、交叉、变异
            new_population = []
            for _ in range(population_size):
                # 选择
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                
                # 交叉
                child = self.crossover(parent1, parent2)
                
                # 变异
                child = self.mutate(child, bounds)
                
                new_population.append(child)
            
            population = new_population
        
        return best_individual if best_individual else [2.0, 0.1, 0.05]
    
    def tournament_selection(self, population, fitness_scores, tournament_size=3):
        """锦标赛选择"""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return population[winner_idx]
    
    def crossover(self, parent1, parent2):
        """交叉操作"""
        crossover_point = random.randint(1, len(parent1) - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child
    
    def mutate(self, individual, bounds, mutation_rate=0.1):
        """变异操作"""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                mutated[i] = random.uniform(bounds[i][0], bounds[i][1])
        return mutated 