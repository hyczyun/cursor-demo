import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.optimize import minimize
import random

class EnergyOptimizer:
    """能耗优化算法类"""
    
    def __init__(self):
        self.best_solution = None
        self.optimization_history = []
        
    def objective_function(self, x, energy_weight, comfort_weight, cost_weight):
        """目标函数：能耗、舒适度、成本的加权组合"""
        # x[0]: 压缩机频率 (30-60 Hz)
        # x[1]: 冷却水流量 (80-150 m³/h)
        # x[2]: 温度设定点 (18-26 °C)
        # x[3]: 启停策略 (0-1)
        
        compressor_freq = x[0]
        water_flow = x[1]
        temp_setpoint = x[2]
        on_off_strategy = x[3]
        
        # 能耗计算
        energy_consumption = (
            0.8 * compressor_freq**2 + 
            0.2 * water_flow**1.5 + 
            0.1 * abs(temp_setpoint - 22)**2
        ) * (0.8 + 0.2 * on_off_strategy)
        
        # 舒适度计算（基于温度偏差）
        comfort_score = 100 - 10 * abs(temp_setpoint - 22)
        
        # 成本计算（基于能耗和运行时间）
        cost = energy_consumption * 0.8 + on_off_strategy * 20
        
        # 加权目标函数（最小化）
        total_objective = (
            energy_weight * energy_consumption / 1000 +
            comfort_weight * (100 - comfort_score) / 100 +
            cost_weight * cost / 1000
        )
        
        return total_objective
    
    def genetic_algorithm(self, energy_weight, comfort_weight, cost_weight, 
                         population_size=50, generations=100):
        """遗传算法优化"""
        # 参数范围
        bounds = [
            (30, 60),    # 压缩机频率
            (80, 150),   # 冷却水流量
            (18, 26),    # 温度设定点
            (0, 1)       # 启停策略
        ]
        
        best_fitness = float('inf')
        best_individual = None
        fitness_history = []
        
        # 初始化种群
        population = []
        for _ in range(population_size):
            individual = [
                random.uniform(bounds[i][0], bounds[i][1]) 
                for i in range(len(bounds))
            ]
            population.append(individual)
        
        for generation in range(generations):
            # 计算适应度
            fitness_scores = []
            for individual in population:
                fitness = self.objective_function(
                    individual, energy_weight, comfort_weight, cost_weight
                )
                fitness_scores.append(fitness)
            
            # 找到最佳个体
            min_fitness_idx = np.argmin(fitness_scores)
            if fitness_scores[min_fitness_idx] < best_fitness:
                best_fitness = fitness_scores[min_fitness_idx]
                best_individual = population[min_fitness_idx].copy()
            
            fitness_history.append(best_fitness)
            
            # 选择
            selected = self.tournament_selection(population, fitness_scores, 3)
            
            # 交叉和变异
            new_population = []
            for i in range(0, population_size, 2):
                if i + 1 < population_size:
                    parent1, parent2 = selected[i], selected[i+1]
                    child1, child2 = self.crossover(parent1, parent2)
                    child1 = self.mutate(child1, bounds, 0.1)
                    child2 = self.mutate(child2, bounds, 0.1)
                    new_population.extend([child1, child2])
                else:
                    new_population.append(selected[i])
            
            population = new_population
        
        return best_individual, fitness_history
    
    def tournament_selection(self, population, fitness_scores, tournament_size):
        """锦标赛选择"""
        selected = []
        for _ in range(len(population)):
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmin(tournament_fitness)]
            selected.append(population[winner_idx])
        return selected
    
    def crossover(self, parent1, parent2):
        """交叉操作"""
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2
    
    def mutate(self, individual, bounds, mutation_rate):
        """变异操作"""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                mutated[i] = random.uniform(bounds[i][0], bounds[i][1])
        return mutated
    
    def particle_swarm_optimization(self, energy_weight, comfort_weight, cost_weight,
                                  n_particles=30, n_iterations=100):
        """粒子群优化算法"""
        # 参数范围
        bounds = [
            (30, 60),    # 压缩机频率
            (80, 150),   # 冷却水流量
            (18, 26),    # 温度设定点
            (0, 1)       # 启停策略
        ]
        
        # 初始化粒子
        particles = []
        velocities = []
        personal_best = []
        personal_best_fitness = []
        
        for _ in range(n_particles):
            particle = [
                random.uniform(bounds[i][0], bounds[i][1]) 
                for i in range(len(bounds))
            ]
            particles.append(particle)
            
            velocity = [
                random.uniform(-1, 1) * (bounds[i][1] - bounds[i][0]) * 0.1
                for i in range(len(bounds))
            ]
            velocities.append(velocity)
            
            personal_best.append(particle.copy())
            fitness = self.objective_function(
                particle, energy_weight, comfort_weight, cost_weight
            )
            personal_best_fitness.append(fitness)
        
        # 全局最佳
        global_best_idx = np.argmin(personal_best_fitness)
        global_best = personal_best[global_best_idx].copy()
        global_best_fitness = personal_best_fitness[global_best_idx]
        
        fitness_history = [global_best_fitness]
        
        # PSO迭代
        w = 0.7  # 惯性权重
        c1 = 1.5  # 个体学习因子
        c2 = 1.5  # 社会学习因子
        
        for iteration in range(n_iterations):
            for i in range(n_particles):
                # 更新速度
                for j in range(len(particles[i])):
                    r1, r2 = random.random(), random.random()
                    velocities[i][j] = (w * velocities[i][j] + 
                                       c1 * r1 * (personal_best[i][j] - particles[i][j]) +
                                       c2 * r2 * (global_best[j] - particles[i][j]))
                
                # 更新位置
                for j in range(len(particles[i])):
                    particles[i][j] += velocities[i][j]
                    particles[i][j] = np.clip(particles[i][j], bounds[j][0], bounds[j][1])
                
                # 更新个体最佳
                fitness = self.objective_function(
                    particles[i], energy_weight, comfort_weight, cost_weight
                )
                if fitness < personal_best_fitness[i]:
                    personal_best[i] = particles[i].copy()
                    personal_best_fitness[i] = fitness
                    
                    # 更新全局最佳
                    if fitness < global_best_fitness:
                        global_best = particles[i].copy()
                        global_best_fitness = fitness
            
            fitness_history.append(global_best_fitness)
        
        return global_best, fitness_history
    
    def reinforcement_learning_simulation(self, energy_weight, comfort_weight, cost_weight,
                                        episodes=100, steps_per_episode=50):
        """强化学习模拟（简化版Q-learning）"""
        # 状态空间：温度偏差、负荷水平、时间
        # 动作空间：压缩机频率调整、流量调整、温度设定点调整
        
        # 简化的Q表
        q_table = {}
        learning_rate = 0.1
        discount_factor = 0.9
        epsilon = 0.1
        
        best_reward = float('-inf')
        best_policy = None
        reward_history = []
        
        for episode in range(episodes):
            state = (0, 0, 0)  # 初始状态
            total_reward = 0
            
            for step in range(steps_per_episode):
                # 选择动作
                if random.random() < epsilon:
                    action = random.choice([0, 1, 2, 3])
                else:
                    if state not in q_table:
                        q_table[state] = [0] * 4
                    action = np.argmax(q_table[state])
                
                # 执行动作
                if action == 0:  # 增加压缩机频率
                    new_state = (state[0] + 1, state[1], state[2])
                elif action == 1:  # 减少压缩机频率
                    new_state = (state[0] - 1, state[1], state[2])
                elif action == 2:  # 增加流量
                    new_state = (state[0], state[1] + 1, state[2])
                else:  # 减少流量
                    new_state = (state[0], state[1] - 1, state[2])
                
                # 计算奖励
                reward = -self.objective_function(
                    [45 + new_state[0], 120 + new_state[1], 22 + new_state[2], 0.5],
                    energy_weight, comfort_weight, cost_weight
                )
                
                # 更新Q值
                if state not in q_table:
                    q_table[state] = [0] * 4
                if new_state not in q_table:
                    q_table[new_state] = [0] * 4
                
                q_table[state][action] = (q_table[state][action] + 
                                        learning_rate * (reward + 
                                        discount_factor * max(q_table[new_state]) - 
                                        q_table[state][action]))
                
                state = new_state
                total_reward += reward
            
            reward_history.append(total_reward)
            
            if total_reward > best_reward:
                best_reward = total_reward
                best_policy = q_table.copy()
        
        # 返回最佳策略对应的参数
        best_state = max(q_table.keys(), key=lambda s: max(q_table[s]))
        best_params = [
            45 + best_state[0],  # 压缩机频率
            120 + best_state[1], # 冷却水流量
            22 + best_state[2],  # 温度设定点
            0.5                  # 启停策略
        ]
        
        return best_params, reward_history
    
    def optimize(self, data):
        # 简单模拟：能耗最小化
        best_setting = np.argmin(data['energy_consumption'])
        return best_setting, data.iloc[best_setting] 