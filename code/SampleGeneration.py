# 样本生成
import pandas as pd 
import random 
import numpy as np 
import csv 
from sklearn.model_selection import train_test_split
import multiprocessing
from PyQt5.QtWidgets import *


class SGB():
    def __init__(self, sample_num=10, pallel_num=2):
        self.sample_num = sample_num
        self.pallel_num = pallel_num

    def load_future_scenario(self, raw_file, dyr_file):
        self.raw_file = raw_file
        self.dyr_file = dyr_file 

    def set_system_security_constraint(self, path_name):
        data = pd.read_csv(path_name, header=0, engine='python')
        self.min_frequency = data.loc[0, '最低频率'] 

    def set_load_shedding_location(self, path_name):
        data = pd.read_csv(path_name, header=0, engine='python')
        loads_shedding = data['负荷'].values.tolist() 
        for i in range(len(loads_shedding)):
            loads_shedding[i] = eval(loads_shedding[i])
        max_percent = data['最大切除比例'].values.tolist()
        self.max_percent = max_percent
        self.loads_shedding = loads_shedding 

    def set_blocking_hvdc_location(self, path_name):
        data = pd.read_csv(path_name, header=0, engine='python')
        data = data['直流'].values.tolist()
        
        for i in range(len(data)):
            data[i] = eval(data[i])
        self.hvdc_block = data  

    def get_shedding_percent(self):
        shedding_percent = np.zeros((self.sample_num, len(self.loads_shedding)))
        for i in range(self.sample_num):
            for j in range(len(self.loads_shedding)):
                shedding_percent[i, j] = random.uniform(0, self.max_percent[j])
        return shedding_percent

    def generate_load_shedding_sample_with_parallel_method(self, simulation_pars):
        p = multiprocessing.Pool(processes=self.pallel_num) 
        sample_feature = p.map(self.generate_load_shedding_sample, simulation_pars)
        p.close()
        p.join()   
        return sample_feature    

    def generate_load_shedding_sample(self, simulation_pars):
        from stepspy import STEPS
        sample_num = simulation_pars['sample_num']
        shedding_percent = simulation_pars['shedding_percent']

        simulator = STEPS(is_default=False, log_file='.\simulation\log_{}.txt'.format(sample_num))
        simulator.load_powerflow_data(self.raw_file, 'PSS/E')  
        simulator.load_dynamic_data(self.dyr_file, 'PSS/E')
        simulator.solve_powerflow('PQ')
 
        buses = simulator.get_all_buses()
        for bus in buses:
            AREA = simulator.get_bus_data(bus, 'I', 'AREA')
            if AREA == 37:  
                simulator.prepare_bus_meter(bus, 'FREQUENCY IN HZ')    
    
        simulator.set_dynamic_simulation_time_step(0.002)
        simulator.set_dynamic_simulator_output_file('.\\simulation\\sample_{}'.format(sample_num))
        simulator.start_dynamic_simulation()
        simulator.run_dynamic_simulation_to_time(0.5)
        
        for hvdc in self.hvdc_block:
            simulator.manually_block_hvdc(hvdc)  
            simulator.trip_fixed_shunt((hvdc[1], '1')) 
        
        simulator.run_dynamic_simulation_to_time(0.6)
        for i in range(len(self.loads_shedding)):
            simulator.scale_load(self.loads_shedding[i], -1*shedding_percent[i])
        
        simulator.run_dynamic_simulation_to_time(5.0)  
        simulator.stop_dynamic_simulation() 

        min_frequency = self.get_min_frequency('.\\simulation\\sample_{}.csv'.format(sample_num))
        if min_frequency > self.min_frequency:
            return [sample_num, min_frequency, 1]
        else:
            return [sample_num, min_frequency, 0] 

    def get_min_frequency(self, sample_path):
        sample_data = pd.read_csv(sample_path, header=0, engine='python')
        columns = list(sample_data)
        frequency_column = []
        for column in columns:
            if 'FREQUENCY' in column:
                frequency_column.append(column)
            else:
                pass
        frequency_data = sample_data.loc[:, frequency_column]
        min_frequency = np.min(frequency_data.values)  # 最低频率
        return min_frequency
         

class SGTL():
    def __init__(self):
        self.sample_scale = 0.01
        self.min_frequency_set = 49.5
        self.__parameter = {'偏移系数': 0.008, '样本数': 100, '并行数': 5}
 
    def set_system_security_constraint(self, path_name):
        data = pd.read_csv(path_name, header=0, engine='python')
        self.min_frequency = data.loc[0, '最低频率']
  
    def load_future_scenario(self, raw_file, dyr_file):
        self.raw_file = raw_file
        self.dyr_file = dyr_file 

    def set_parameter_data(self, par_name, value):
        self.__parameter[par_name] = value

    def set_best_scheme(self, best_scheme):
        self.best_scheme = best_scheme

    def set_load_shedding_location(self, path_name):
        data = pd.read_csv(path_name, header=0, engine='python')
        loads_shedding = data['负荷'].values.tolist() 
        for i in range(len(loads_shedding)):
            loads_shedding[i] = eval(loads_shedding[i])
        max_percent = data['最大切除比例'].values.tolist()
        self.max_percent = max_percent
        self.loads_shedding = loads_shedding

    def set_blocking_hvdc_location(self, path_name):
        data = pd.read_csv(path_name, header=0, engine='python')
        data = data['直流'].values.tolist()
        
        for i in range(len(data)):
            data[i] = eval(data[i])
        self.hvdc_block = data  

    def get_min_frequency(self, sample_path):
        sample_data = pd.read_csv(sample_path, header=0, engine='python')
        columns = list(sample_data)
        frequency_column = []
        for column in columns:
            if 'FREQUENCY' in column:
                frequency_column.append(column)
            else:
                pass

        frequency_data = sample_data.loc[:, frequency_column]
        min_frequency = np.min(frequency_data.values)  # 最低频率
        return min_frequency

    def generate_new_scenario_load_shedding_sample(self):
        '''生成新方式下的切负荷比例'''
        best_inds = pd.read_csv(self.best_scheme, header=0, engine='python')
        best_inds = best_inds.values
        if self.__parameter['样本数'] < best_inds.shape[0]:
            sample_num = self.__parameter['样本数']
        else:
            sample_num = best_inds.shape[0]
        
        div = self.__parameter['偏移系数']
        shedding_percent = np.zeros((sample_num, best_inds.shape[1]))
        for i in range(sample_num):
            for j in range(shedding_percent.shape[1]):
                if self.__parameter['方向'] == '增加':
                    c = random.uniform(best_inds[i, j], best_inds[i, j]+div)
                elif self.__parameter['方向'] == '减少':
                    c = random.uniform(best_inds[i, j]-div, best_inds[i, j])
                else:
                    c = 0.0

                if c < 0.0:
                    c = 0.0
                if c > 0.2:
                    c = 0.2
                shedding_percent[i, j] = c

        return np.array(shedding_percent)

    def generate_load_shedding_sample_with_parallel_method(self, simulation_pars):
        p = multiprocessing.Pool(processes=self.__parameter['并行数']) 
        sample_feature = p.map(self.generate_load_shedding_sample, simulation_pars)
        p.close()
        p.join()     
        return sample_feature

    def generate_load_shedding_sample(self, simulation_pars):
        from stepspy import STEPS
        sample_num = simulation_pars['sample_num']
        scale_percent = simulation_pars['scale_percent']

        simulator = STEPS(is_default=False, log_file='.\simulation\log_{}.txt'.format(sample_num))
        simulator.load_powerflow_data(self.raw_file, 'PSS/E')  
        simulator.load_dynamic_data(self.dyr_file, 'PSS/E')
        simulator.solve_powerflow('PQ')
 
        buses = simulator.get_all_buses()
        for bus in buses:
            AREA = simulator.get_bus_data(bus, 'I', 'AREA')
            if AREA == 37:  
                simulator.prepare_bus_meter(bus, 'FREQUENCY IN HZ')    
    
        simulator.set_dynamic_simulation_time_step(0.002)
        simulator.set_dynamic_simulator_output_file('.\\simulation\\sample_{}'.format(sample_num))
        simulator.start_dynamic_simulation()
        simulator.run_dynamic_simulation_to_time(0.5)
        
        for hvdc in self.hvdc_block:
            simulator.manually_block_hvdc(hvdc)  
            simulator.trip_fixed_shunt((hvdc[1], '1')) 
        
        simulator.run_dynamic_simulation_to_time(0.6)
        for i in range(len(self.loads_shedding)):
            simulator.scale_load(self.loads_shedding[i], -1*scale_percent[i])
        
        simulator.run_dynamic_simulation_to_time(5.0)  
        simulator.stop_dynamic_simulation() 

        min_frequency = self.get_min_frequency('.\\simulation\\sample_{}.csv'.format(sample_num))
        if min_frequency > self.min_frequency:
            return [sample_num, min_frequency, 1]
        else:
            return [sample_num, min_frequency, 0]
    
    