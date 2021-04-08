import numpy as np 
import pandas as pd 
import random
import multiprocessing
import csv
import matplotlib.pyplot as plt 


def generate_load_trip_sample_with_parallel_method():
    '''
    Parallel simulation to generate samples.
    Args: None.
    Rets: None.
    '''
    loads_trip_percent = pd.read_csv('.\\feature\loads_trip_percent.csv', header=0)
    loads_trip_percent = loads_trip_percent.values

    simulation_pars = []
    for i in range(loads_trip_percent.shape[0]):
        par = {'sample_num': i+1, 'scale_percent': loads_trip_percent[i, :]}
        simulation_pars.append(par)
    
    p = multiprocessing.Pool(processes=20)  
    res = p.map(generate_load_trip_sample, simulation_pars)
    p.close()
    p.join()
    return

def generate_load_trip_sample(simulation_pars):
    '''
    generate load trip sample. At present, the simulation operation is only for frequency. 
    Args: simulation_pars, number and trip percent of simulation.
    Rets: None.  
    '''
    from stepspy import STEPS
    sample_num = simulation_pars['sample_num']
    scale_percent = simulation_pars['scale_percent']
   
    simulator = STEPS(is_default=False, log_file='.\log\log_{}.txt'.format(sample_num))
    simulator.set_allowed_maximum_bus_number(100000)
    simulator.load_powerflow_data('.\\data\\shandong_power_grid_level_102.raw', 'PSS/E')  
    simulator.load_dynamic_data('.\\data\\bench_shandong_change_with_gov.dyr', 'PSS/E')
    simulator.solve_powerflow('NR')
    
    buses = simulator.get_all_buses()
    loads = simulator.get_all_loads()
    generators = simulator.get_all_generators()
    lines = simulator.get_all_lines()
    
    for bus in buses:
        AREA = simulator.get_bus_data(bus, 'I', 'AREA')
        if AREA == 37:  # Monitor the frequency and voltage of all buses in Shandong power grid
            simulator.prepare_bus_meter(bus, 'FREQUENCY IN HZ')    
            #simulator.prepare_bus_meter(bus, 'VOLTAGE IN PU')
    '''        
    for generator in generators:
        AREA = simulator.get_bus_data(generator[0], 'I', 'AREA')
        if AREA == 37:  # Monitor the power angle of all generators in Shandong Power Grid
            simulator.prepare_generator_meter(generator, 'ROTOR ANGLE IN DEG')

    for line in lines:
        VBASE_KV = simulator.get_bus_data(line[0], 'F', 'VBASE_KV')
        if int(VBASE_KV) == 1050:  #  Monitor the active power flow of UHV AC line of Shandong Power Grid
            simulator.prepare_line_meter(line, 'ACTIVE POWER IN MW', 'SENDING')
    '''        
    simulator.set_dynamic_simulation_time_step(0.002)
    simulator.set_dynamic_simulator_output_file('.\\simulation_result\\sample_{}'.format(sample_num))
    simulator.start_dynamic_simulation()
    simulator.run_dynamic_simulation_to_time(1.0)
    # Block two DC lines at 0.0s, loss of 8000mw power.
    simulator.manually_block_hvdc((201, 61, 'ZQH'))  
    simulator.manually_block_hvdc((201, 57, 'ZQL'))   
    simulator.trip_fixed_shunt((61, '1'))  
    simulator.trip_fixed_shunt((57, '1')) 
    
    # After 100 ms, emergency load shedding measures for specific accidents are taken
    simulator.run_dynamic_simulation_to_time(1.1)
    for i in range(len(loads)):
        simulator.scale_load(loads[i], -1*scale_percent[i])
    
    # Simulation ends in 10s
    simulator.run_dynamic_simulation_to_time(10.0)  
    simulator.stop_dynamic_simulation() 
    print('Successfully generate sample {} frequency response'.format(sample_num))
    return

def generate_random_load_trip_percent(sample_num):
    '''
    generate random number of load cutting, and maximum cutting is 20%.
    Args:
        sample_num, int, sample number.
    Rets: None
    '''
    from stepspy import STEPS 

    raw_file = '.\\data\\bench_shandong_change_with_gov.raw'    
    simulator = STEPS(is_default=False, log_file='log.txt')
    simulator.set_allowed_maximum_bus_number(100000)
    simulator.load_powerflow_data(raw_file, 'PSS/E') 
    simulator.solve_powerflow('NR')
    loads = simulator.get_all_loads()
    
    samples_input = LHSampling(len(loads), [0, 0.2], 5000)
    with open('.\\feature\loads_trip_percent.csv', 'w', newline='') as f: 
        csv_write = csv.writer(f)
        csv_write.writerow(list(loads))
        csv_write.writerows(samples_input)     
    return


def LHSampling(Parameters_num, bounds, N): 
    '''
    Latin square sampling procedure
    Args:
        1) Parameters_num, int, Single sample data dimension.
    '''
    result = np.empty([N, Parameters_num]) 
    temp = np.empty([N])
    d = 1.0 / N
 
    for i in range(Parameters_num):
        for j in range(N):
            temp[j] = np.random.uniform(low=j * d, high=(j + 1) * d, size = 1)[0]
        np.random.shuffle(temp) ##
        for j in range(N):
            result[j, i] = temp[j]
   
    lower_bounds = bounds[0]
    upper_bounds = bounds[1]
    if np.any(lower_bounds > upper_bounds):
        print('bounds_error')
        return None
    #   sample * (upper_bound - lower_bound) + lower_bound
    result = lower_bounds + result * (upper_bounds - lower_bounds)
    return result

def check_a_load_shedding_scheme():
    '''
    棢�查某个切负荷方案
    '''
    from stepspy import STEPS
    loads_trip_percent = pd.read_csv('.\\feature\loads_trip_percent.csv', header=0)
    loads_trip_percent = loads_trip_percent.values

    sample_num = 3150
    scale_percent = loads_trip_percent[3149, :]
   
    simulator = STEPS(is_default=False, log_file='.\log\log_{}.txt'.format(sample_num))
    simulator.set_allowed_maximum_bus_number(100000)
    simulator.load_powerflow_data('.\\data\\shandong_power_grid_level_100.raw', 'PSS/E')  
    simulator.load_dynamic_data('.\\data\\bench_shandong_change_with_gov.dyr', 'PSS/E')
    simulator.solve_powerflow('NR')
    
    buses = simulator.get_all_buses()
    loads = simulator.get_all_loads()
    generators = simulator.get_all_generators()
    lines = simulator.get_all_lines()
    
    for bus in buses:
        AREA = simulator.get_bus_data(bus, 'I', 'AREA')
        if AREA == 37:  # Monitor the frequency and voltage of all buses in Shandong power grid
            simulator.prepare_bus_meter(bus, 'FREQUENCY IN HZ')    
            #simulator.prepare_bus_meter(bus, 'VOLTAGE IN PU')
            
    for generator in generators:
        AREA = simulator.get_bus_data(generator[0], 'I', 'AREA')
        if AREA == 37:  # Monitor the power angle of all generators in Shandong Power Grid
            simulator.prepare_generator_meter(generator, 'ROTOR ANGLE IN DEG')

    for line in lines:
        VBASE_KV = simulator.get_bus_data(line[0], 'F', 'VBASE_KV')
        if int(VBASE_KV) == 1050:  #  Monitor the active power flow of UHV AC line of Shandong Power Grid
            simulator.prepare_line_meter(line, 'ACTIVE POWER IN MW', 'SENDING')
            
    simulator.set_dynamic_simulation_time_step(0.002)
    simulator.set_dynamic_simulator_output_file('.\\simulation_result\\sample_{}'.format(sample_num))
    simulator.start_dynamic_simulation()
    simulator.run_dynamic_simulation_to_time(1.0)
    # Block two DC lines at 0.0s, loss of 8000mw power.
    simulator.manually_block_hvdc((201, 61, 'ZQH'))  
    simulator.manually_block_hvdc((201, 57, 'ZQL'))   
    simulator.trip_fixed_shunt((61, '1'))  
    simulator.trip_fixed_shunt((57, '1')) 
    
    # After 100 ms, emergency load shedding measures for specific accidents are taken
    simulator.run_dynamic_simulation_to_time(1.1)
    for i in range(len(loads)):
        simulator.scale_load(loads[i], -1*scale_percent[i])
    
    # Simulation ends in 10s
    simulator.run_dynamic_simulation_to_time(10.0)  
    simulator.stop_dynamic_simulation()
    return



if __name__=='__main__':
    #generate_random_load_trip_percent(5000)
    generate_load_trip_sample_with_parallel_method()
    #check_a_load_shedding_scheme()