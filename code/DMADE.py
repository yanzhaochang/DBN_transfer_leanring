import numpy as np
import pandas as pd 
import random
import multiprocessing
import csv
import math 

from PyQt5.QtWidgets import *

from keras.models import load_model
import matplotlib.pyplot as plt

from sklearn.externals import joblib


class ADELSM():  # 代理辅助模型驱动的差分进化切负荷模型
    def __init__(self):
        self.__size = 100
        self.__iter_num = 300 
        self.__F = 0.8
        self.__CR = 0.4

        self.__process = []
        self.__constraint = {'最低频率': 49.55, '频率下界': 49.5, '频率上界': 49.6}

        self.set_load_shedding_location('.\\参数设置\\切负荷站.csv')
        self.set_blocking_hvdc_location('.\\参数设置\\闭锁直流.csv')
 
    def set_load_shedding_location(self, path_name):
        data = pd.read_csv(path_name, header=0, engine='python')
        loads_shedding = data['负荷'].values.tolist() 
        for i in range(len(loads_shedding)):
            loads_shedding[i] = eval(loads_shedding[i])

        self.max_percent = data['最大切除比例'].values.tolist()
        self.loads_shedding = loads_shedding
        self.__dim = len(self.loads_shedding)  # 个体特征维数
        self.x_min = np.zeros(self.__dim)
        self.x_max = np.array(self.max_percent) 

    def set_blocking_hvdc_location(self, path_name):
        data = pd.read_csv(path_name, header=0, engine='python')
        data = data['直流'].values.tolist()
        for i in range(len(data)):
            data[i] = eval(data[i])
        self.hvdc_block = data  

    # 以下为加载代理辅助模型，目前只有频率模型
    def load_frequency_classification_prediction_model(self, file):
        self.frequency_classification_model = load_model(file)  

    # 以下为加载运行场景
    def load_scenario_data(self, raw_file, dyr_file):
        from stepspy import STEPS 
        self.simulator = STEPS(is_default=False, log_file='.\\log\\log_op.txt')
        self.simulator.load_powerflow_data(raw_file, 'PSS/E')
        self.simulator.load_dynamic_data(dyr_file, 'PSS/E')
        self.simulator.solve_powerflow('PQ')
        self.__loads_p = np.zeros(len(self.loads_shedding))  # 该方式下的负荷量
        for i in range(len(self.loads_shedding)):
            self.__loads_p[i] = self.simulator.get_load_data(self.loads_shedding[i], 'F', 'PP0_MW')        
        return


    # 以下为设置差分进化算法的参数
    def set_evolution_parameter(self, par_name, value):
        if par_name == '种群数目':
            self.__size = int(value)
        elif par_name == '进化代数':
            self.__iter_num = int(value)

        elif par_name == '交叉因子':
            self.__CR = value    

        elif par_name == '变异因子':
            self.__F = value    

        elif par_name == '历史方案':
            if len(value) == 0:
                return 
            else:
                best_ind = pd.read_csv(value, header=0, engine='python')
                best_ind = best_ind.loc[0].values 

                par = pd.read_csv('.\\参数设置\\模型更新样本参数.csv', header=0, engine='python')
                scale = par.loc[0, '分布系数']
                value = 2.5758 * scale

                for i in range(self.__dim):
                    lower_limit = best_ind[i] - value  # 取99%置信度区间
                    upper_limit = best_ind[i] + value
                    if lower_limit > 0.0:
                        self.x_min[i] = lower_limit
                    if upper_limit < self.max_percent[i]:
                        self.x_max[i] = upper_limit 
        else:
            pass

    #以下为种群初始化
    def initialize_population(self):
        self.population = np.zeros((self.__size, self.__dim))
        for i in range(self.__dim):
            for j in range(self.__size):  # 初始种群在限度范围内随机初始化
                self.population[j, i] = random.uniform(self.x_min[i], self.x_max[i]) 

        self.population_fitness = np.zeros(self.__size)
        for i in range(self.__size):
            self.population_fitness[i] = self.get_individual_fitness(self.population[i, :])    

    # 以下为进化过程
    def operate_evolution(self):
        '''实施进化计算'''
        k = 0
        while True:
            mutation_population = self.operate_population_mutation()
            process_population = self.operate_population_crossover(mutation_population)
            self.operate_selection(process_population)
            for j in range(self.__size):
                self.population_fitness[j] = self.get_individual_fitness(self.population[j, :])             
            mean_fitness = np.mean(self.population_fitness)
            max_fitness = np.max(self.population_fitness)
            min_fitness = np.min(self.population_fitness)
    
            self.process_output.append('第{}代-平均值{}-最大值{}-最小值{}'.format(k, int(mean_fitness), int(max_fitness), int(min_fitness))) 
            self.process_output.verticalScrollBar().setValue(self.process_output.verticalScrollBar().maximum())
            QApplication.processEvents()
            k = k + 1
            if k > self.__iter_num:
                break 

        for i in range(self.__size):
            self.population_fitness[i] = self.get_individual_fitness(self.population[i, :])     
        return 

    def operate_evolution_one_enpouch(self):
        mutation_population = self.operate_population_mutation()
        process_population = self.operate_population_crossover(mutation_population)
        self.operate_selection(process_population)
        for j in range(self.__size):
            self.population_fitness[j] = self.get_individual_fitness(self.population[j, :])    
        mean_fitness = np.mean(self.population_fitness)
        max_fitness = np.max(self.population_fitness)
        min_fitness = np.min(self.population_fitness)      
        return int(mean_fitness), int(max_fitness), int(min_fitness)

    def operate_population_mutation(self):
        '''对种群实施变异'''
        mutation_population = np.zeros((self.__size, self.__dim))
        for i in range(self.__size):
            list_num = list(range(0, self.__size, 1))
            list_num.remove(i)
            res = random.sample(list_num, 3)
            mutation_individual = self.population[res[0], :] + self.__F * (self.population[res[1], :] - self.population[res[2], :])  # 变异操作，产生新个体
            
            for j in range(self.__dim):  # 特征越限处理
                if mutation_individual[j] < self.x_min[j] or mutation_individual[j] > self.x_max[j]:
                    mutation_individual = self.x_min + random.random() * (self.x_max - self.x_min)
                    break 
            
            mutation_population[i, :] = mutation_individual 
        return mutation_population

    def operate_population_crossover(self, mutation_population):
        '''进行交叉操作'''
        process_population = np.zeros((self.__size, self.__dim))
        for i in range(self.__size):
            randn = random.randint(0, self.__dim)
            for j in range(self.__dim):
                rand_float = random.random()
                if rand_float <= self.__CR or randn == j:
                    process_population[i, j] = mutation_population[i, j]
                else:
                    process_population[i, j] = self.population[i, j]
        return process_population
 

    def operate_selection(self, process_population):
        '''对个体进行选择和更新'''
        for i in range(self.__size):
            ind_1 = self.population[i, :]
            ind_2 = process_population[i, :]
            better_ind = self.select_individual_cla(ind_1, ind_2)
            self.population[i, :] = better_ind
        return

    def select_individual_cla(self, ind_1, ind_2):
        ind_1_cla = self.frequency_classification_model.predict_classes(5 * ind_1.reshape((1, -1)))
        ind_1_cla = ind_1_cla[0, 0]        
        ind_2_cla = self.frequency_classification_model.predict_classes(5 * ind_2.reshape((1, -1)))
        ind_2_cla = ind_2_cla[0, 0] 
        
        if ind_1_cla == 1 and ind_2_cla == 1:
            fit_1 = self.get_individual_fitness(ind_1)
            fit_2 = self.get_individual_fitness(ind_2)
            if fit_1 > fit_2:
                better_ind = ind_2
            else:
                better_ind = ind_1
        elif ind_1_cla == 0 and ind_2_cla == 1: 
            better_ind = ind_2

        elif ind_1_cla == 0 and ind_2_cla == 0:
            better_ind = np.zeros(self.__dim)
            for i in range(self.__dim):
                better_ind[i] = random.uniform(self.x_min[i], self.x_max[i])
        else:
            better_ind = ind_1
        return  better_ind               

    # 以下为个体适应度计算函数
    def get_individual_fitness(self, individual):  # individual应该是个一维数组
        value = np.sum(individual * self.__loads_p)  # 切负荷总量
        return value   

    def save_best_individual(self, file):
        min_index = np.argmin(self.population_fitness)
        best_individual = self.population[min_index, :]
        with open(file, 'w', newline='') as f: 
            csv_write = csv.writer(f)
            csv_write.writerow(self.loads_shedding)
            csv_write.writerow(best_individual)

        loads_name = []
        for load in self.loads_shedding:
            NAME = self.simulator.get_bus_data(load[0], 'S', 'NAME')
            loads_name.append(NAME)
        return best_individual, self.__loads_p, self.loads_shedding, loads_name

    def check_evolution_result(self, best_individual):
        buses = self.simulator.get_all_buses() 
        for bus in buses:
            AREA = self.simulator.get_bus_data(bus, 'I', 'AREA')
            if AREA == 37:  
                self.simulator.prepare_bus_meter(bus, 'FREQUENCY IN HZ')          
        self.simulator.set_dynamic_simulation_time_step(0.002)
        self.simulator.set_dynamic_simulator_output_file('.\\优化结果\\代理辅助模型校验结果')
        self.simulator.start_dynamic_simulation()
        self.simulator.run_dynamic_simulation_to_time(0.5)
    
        for hvdc in self.hvdc_block:
            self.simulator.manually_block_hvdc(hvdc)  
            self.simulator.trip_fixed_shunt((hvdc[1], '1')) 

        self.simulator.run_dynamic_simulation_to_time(0.6)
        for i in range(len(self.loads_shedding)):
            self.simulator.scale_load(self.loads_shedding[i], -1*best_individual[i])
    
        self.simulator.run_dynamic_simulation_to_time(5.0)  
        self.simulator.stop_dynamic_simulation() 

        sample_data = pd.read_csv('.\\优化结果\\代理辅助模型校验结果.csv', header=0, engine='python')
        columns = list(sample_data)
        frequency_column = []
        for column in columns:
            if 'FREQUENCY' in column:
                frequency_column.append(column)
            else:
                pass
        frequency_data = sample_data.loc[:, frequency_column]
        min_frequency = np.min(frequency_data.values)
        return min_frequency
        


    
