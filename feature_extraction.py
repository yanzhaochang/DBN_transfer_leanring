'''
The improved binary table method is used to evaluate the safety of frequency and voltage.
For frequency, [fcr, Tcr]. Change it into ita_f = [f_ext - (fcr - k * Tcr)] * 100%. The setting for frequency is [49.7, 0.3] 
For voltage, [Vcr, Tcr].
'''
import numpy as np 
import pandas as pd  
import csv 

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）



def generate_all_sample_transient_index():
    '''
    The transient calculation indexes of all samples are generated.
    '''
    with open('.\\feature\\sample_feature.csv', 'w', newline='') as f: 
        csv_write = csv.writer(f)
        csv_write.writerow(['frequency', 'frequency_margin', 'frequency_offset'])


    with open('.\\feature\\sample_feature.csv', 'a+', newline='') as f:
        for i in range(5000):
            sample_index = get_sample_transient_index('.\\simulation_result\\sample_{}.csv'.format(i+1))
            print('Successfully extract sample {} transient index'.format(i+1))
            csv_write = csv.writer(f)
            csv_write.writerow(sample_index) 
    return

def get_sample_transient_index(file):
    '''
    Get the characteristics of a sample. 
    The indexes include minimum frequency, minimum voltage and maximum power angle difference.
    Now we will use system minimum frequency to test transfer learning model.
    '''
    sample_data = pd.read_csv(file, header=0)
    columns = list(sample_data)

    frequency_column = []
    voltage_column = []
    angle_column = []
    
    for column in columns:
        if 'FREQUENCY' in column:
            frequency_column.append(column)
        elif 'VOLTAGE' in column:
            voltage_column.append(column)
        elif 'ANGLE' in column:
            angle_column.append(column)
        else:
            pass

    frequency_data = sample_data.loc[:, frequency_column]
    #voltage_data = sample_data.loc[:, voltage_column]
    #rotor_angle_data = sample_data.loc[:, angle_column]

    min_frequency = get_sample_min_frequency(frequency_data)
    #min_voltage = get_sample_min_voltage(voltage_data)
    #max_angle_diff = get_sample_max_rotor_different_angle(rotor_angle_data)

    system_frequency_safety_margin = get_sample_frequency_margin(frequency_data)
    return [min_frequency, system_frequency_safety_margin, 50.0-min_frequency]

def get_sample_frequency_margin(frequency_data):
    (m, n) = frequency_data.shape
    system_frequency_index = np.zeros(n)
    for i in range(n):
        curve = frequency_data.iloc[:, i]
        frequency_index = get_frequency_curve_index(curve)
        system_frequency_index[i] = frequency_index

    system_frequency_safety_margin = np.min(system_frequency_index)
    return system_frequency_safety_margin

def get_frequency_curve_index(curve):
    '''
    计算单根频率曲线的裕度指标
    '''
    Fcr = 49.75
    Tcr = 0.500
    curve = curve.values
    high_frequency = np.max(curve) 
    low_frequency = np.min(curve) 
    for i in range(20):
        middle_frequency = 0.5 * (high_frequency + low_frequency)
        lower_than_middle = [x for x in curve if x < middle_frequency]
        T_middle = len(lower_than_middle) * 0.002
        F_middle = middle_frequency
        if round(T_middle, 3) == Tcr:
            break
        elif T_middle < Tcr:
            low_frequency = middle_frequency
        elif T_middle > Tcr:
            high_frequency = middle_frequency
        else:
            pass   
    frequency_index = (F_middle - Fcr) / Fcr * 100
    return frequency_index


def get_sample_min_frequency(frequency_data):
    '''
    Get the lowest frequency of the system buses.
    '''
    frequency_data = frequency_data.values 
    min_frequency = np.min(frequency_data)
    return min_frequency

def get_sample_min_voltage(voltage_data):
    '''
    Get the lowest voltage of the system buses.
    '''
    voltage_data = voltage_data.values 
    min_voltage = np.min(voltage_data)
    return min_voltage

def get_sample_line_max_active_power(line_active_power_data):
    return


def get_sample_max_rotor_different_angle(rotor_angle_data):
    '''
    Obtain the maximum power angle difference of generator.
    '''
    generator2_angle = rotor_angle_data['ROTOR ANGLE IN DEG @ GENERATOR 1 AT BUS 102'].values
    rotor_angle_data = rotor_angle_data.values
    angle_dif_data = rotor_angle_data - generator2_angle.reshape((-1, 1))
    max_angle_diff = np.max(np.abs(angle_dif_data))
    return max_angle_diff

def normalize_load_trip_percent():
    '''
    The feature is normalized and reduced to 0-1.
    '''
    percent = pd.read_csv('.\\feature\loads_trip_percent.csv', header=0)
    min_max_scaler = MinMaxScaler()
    feature = min_max_scaler.fit_transform(percent.values)
    with open('.\\feature\\normalization_feature.csv', 'w', newline='') as f: 
        csv_write = csv.writer(f)
        csv_write.writerows(feature.tolist())  
    return

def split_sample():
    '''
    The samples are divided into training set and test set, where test sets is 20%.
    '''
    sample_data = pd.read_csv('.\\feature\\normalization_feature.csv', header=None)
    sample_data = sample_data.values

    sample_feature = pd.read_csv('.\\feature\\sample_feature.csv', header=0)
    sample_feature = sample_feature[['frequency']].values

    x_train, x_test, y_train, y_test = train_test_split(sample_data, sample_feature, test_size=0.2, random_state=10)   
    
    with open('.\\feature\\x_train.csv', 'w', newline='') as f: 
        csv_write = csv.writer(f)
        csv_write.writerows(x_train.tolist())

    with open('.\\feature\\x_test.csv', 'w', newline='') as f: 
        csv_write = csv.writer(f)
        csv_write.writerows(x_test.tolist())

    with open('.\\feature\y_train.csv', 'w', newline='') as f: 
        csv_write = csv.writer(f)
        csv_write.writerows(y_train.tolist())

    with open('.\\feature\y_test.csv', 'w', newline='') as f: 
        csv_write = csv.writer(f)
        csv_write.writerows(y_test.tolist())         
    return

if __name__=='__main__':
    generate_all_sample_transient_index()
    
    normalize_load_trip_percent()
    
    split_sample()
    
    