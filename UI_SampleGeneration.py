from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

import csv 
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
import sys 
sys.path.append('.\\code')
from SampleGeneration import SGB, SGTL 



class UI_SGBM(QDialog):
    message_singal = pyqtSignal(str)
    def __init__(self):
        super(UI_SGBM, self).__init__()
        self.setWindowTitle('基础模型训练样本生成')
        self.resize(700, 500)
        self.setWindowIcon(QIcon('.\\data\\safety.png'))

        self.mainlayout = QGridLayout()
        self.setLayout(self.mainlayout)
        
        self.parameter_group = QGroupBox('参数设置')
        self.process_group = QGroupBox('生成过程')

        self.mainlayout.addWidget(self.parameter_group, 0, 0)
        self.mainlayout.addWidget(self.process_group, 0, 1)
        
        self.init_parameter()
        self.init_process()
        
    def init_parameter(self):
        self.parameter_layout = QGridLayout()
        self.parameter_group.setLayout(self.parameter_layout)        

        scenario_label = QLabel('目标场景')
        scenario_button = QPushButton('选择')
        scenario_button.clicked.connect(self.choose_scenario)
        self.scenario_line = QLineEdit()
        self.scenario_line.setText('.\\scenarios\\example_scenario_source.raw')
        self.parameter_layout.addWidget(scenario_label, 0, 0)
        self.parameter_layout.addWidget(self.scenario_line, 0, 1, 1, 4)  
        self.parameter_layout.addWidget(scenario_button, 0, 5)  
        
        label_sample_num = QLabel('样本数')
        self.sample_num_spinbox = QSpinBox()
        self.sample_num_spinbox.setMinimum(10)
        self.sample_num_spinbox.setMaximum(10000)
        self.sample_num_spinbox.setValue(100)
        self.sample_num_spinbox.setSingleStep(100)        

        pallel_num_label = QLabel('并行数')
        self.pallel_num_spinbox = QSpinBox()
        self.pallel_num_spinbox.setMinimum(1)
        self.pallel_num_spinbox.setMaximum(20)
        self.pallel_num_spinbox.setValue(5)
        self.pallel_num_spinbox.setSingleStep(1)

        self.parameter_layout.addWidget(label_sample_num, 1, 0)
        self.parameter_layout.addWidget(self.sample_num_spinbox, 1, 1)
        self.parameter_layout.addWidget(QLabel(), 1, 2)

        self.parameter_layout.addWidget(pallel_num_label, 2, 0)
        self.parameter_layout.addWidget(self.pallel_num_spinbox, 2, 1) 

        test_data_percent_label = QLabel('测试集比例') 
        self.test_data_percent_spinbox = QDoubleSpinBox()
        self.test_data_percent_spinbox.setValue(0.2)
        self.test_data_percent_spinbox.setMinimum(0.0)
        self.test_data_percent_spinbox.setMaximum(1.0)
        self.test_data_percent_spinbox.setSingleStep(0.05)   

        self.parameter_layout.addWidget(test_data_percent_label, 3, 0)
        self.parameter_layout.addWidget(self.test_data_percent_spinbox, 3, 1) 
      
        self.button_generation = QPushButton('生成')
        self.parameter_layout.addWidget(self.button_generation, 4, 4)
        self.button_generation.clicked.connect(self.generate_samples)

    def init_process(self):
        self.process_layout = QGridLayout()
        self.process_group.setLayout(self.process_layout)
        
        self.process_table = QTableWidget()
        self.process_table.setColumnCount(3)
        self.process_table.setHorizontalHeaderLabels(['编号', '最低频率', '标签'])
        self.process_table.verticalHeader().setVisible(False)  # 隐藏水平表头  
        self.process_layout.addWidget(self.process_table, 0, 0, 1, 4)

        stable_label = QLabel('稳定样本总数')
        self.stable_line = QLineEdit()
        unstable_label = QLabel('失稳样本总数')
        self.unstable_line = QLineEdit()
        self.process_layout.addWidget(stable_label, 1, 0)
        self.process_layout.addWidget(self.stable_line, 1, 1)
        self.process_layout.addWidget(unstable_label, 1, 2)
        self.process_layout.addWidget(self.unstable_line, 1, 3)

        self.sample_thread = SampleGenerationThread()  #实例化线程对象
        self.sample_thread.finish_signal.connect(self.finish_generation_process)
        self.sample_thread.process_signal.connect(self.refresh_process_information)       
        
    def generate_samples(self):
        self.process_table.clearContents()
        sample_num = self.sample_num_spinbox.value()
        self.process_table.setRowCount(sample_num)
        pallel_num = self.pallel_num_spinbox.value()

        scenario = self.scenario_line.text()
        if len(scenario) == 0:
            dialog = QMessageBox(QMessageBox.Warning, '警告', '未导入电网目标运行方式!!!!')
            dialog.exec_()
            return

        self.sample_thread.set_generation_parameter({'场景': scenario, '样本数': sample_num, '并行数': pallel_num})    
        self.button_generation.setEnabled(False)
        self.sample_thread.start()
        
    def finish_generation_process(self, list_singal):
        self.button_generation.setEnabled(True)
        row = self.process_table.rowCount()
        labels = []
        for i in range(row):
            label = self.process_table.item(i, 2).text()
            labels.append(int(label))
        labels = np.array(labels)
        self.stable_line.setText(str(np.sum(labels)))
        self.unstable_line.setText(str(len(labels) - np.sum(labels)))

        QMessageBox.information(self, "提示", "样本生成完毕!", QMessageBox.Yes, QMessageBox.Yes)
        shedding_percent, sample_feature = list_singal[0], list_singal[1]
        self.spilt_sample(shedding_percent, sample_feature)

    def spilt_sample(self, shedding_percent, sample_feature):
        loads_shedding = self.get_loads_shedding()

        test_data_percent = self.test_data_percent_spinbox.value()
        x_train, x_test, y_train, y_test = train_test_split(shedding_percent, sample_feature, test_size=test_data_percent, random_state=10)   
    
        with open('.\\data\\x_train.csv', 'w', newline='') as f: 
            csv_write = csv.writer(f)
            csv_write.writerow(loads_shedding)
            csv_write.writerows(x_train.tolist())

        with open('.\\data\\x_test.csv', 'w', newline='') as f: 
            csv_write = csv.writer(f)
            csv_write.writerow(loads_shedding)
            csv_write.writerows(x_test.tolist())

        with open('.\\data\\y_train.csv', 'w', newline='') as f: 
            csv_write = csv.writer(f)
            csv_write.writerow(['编号', '频率', '标签'])
            csv_write.writerows(y_train.tolist())

        with open('.\\data\\y_test.csv', 'w', newline='') as f: 
            csv_write = csv.writer(f)
            csv_write.writerow(['编号', '频率', '标签'])
            csv_write.writerows(y_test.tolist())

    def choose_scenario(self):
        openfile_name = QFileDialog.getOpenFileName(self, '场景', '' ,'file(*.raw)')
        if len(openfile_name[0]) == 0:
            return   
        path_name = openfile_name[0]
        if len(path_name) == 0:
            return
        self.scenario_line.setText(path_name)  

    def refresh_process_information(self, sample_feature):
        for i in range(len(sample_feature)):
            self.process_table.setItem(sample_feature[i][0], 0, QTableWidgetItem(str(sample_feature[i][0]))) 
            frequency = round(sample_feature[i][1], 4)
            self.process_table.setItem(sample_feature[i][0], 1, QTableWidgetItem(str(frequency)))
            self.process_table.setItem(sample_feature[i][0], 2, QTableWidgetItem(str(sample_feature[i][2])))
        self.process_table.verticalScrollBar().setValue(sample_feature[0][0])         

    def get_loads_shedding(self):
        data = pd.read_csv('.\\data\\loads_shedding.csv', header=0, engine='python')
        loads_shedding = data['负荷'].values.tolist() 
        for i in range(len(loads_shedding)):
            loads_shedding[i] = eval(loads_shedding[i])
        return loads_shedding

class SampleGenerationThread(QThread):
    finish_signal = pyqtSignal(list)  
    process_signal = pyqtSignal(list)

    def __init__(self):
        super(SampleGenerationThread, self).__init__()
        self.load_shedding_location = '.\\data\\loads_shedding.csv'
        self.blocking_hvdc_location = '.\\data\\hvdc_block.csv'
        self.dyr_file = '.\\data\\bench_shandong_change_with_gov.dyr'
        self.system_security_constraint = '.\\data\\safety_constraint.csv'
         
    def set_generation_parameter(self, parameter):
        self.parameter = parameter
         
    def run(self): #线程执行函数
        model = SGB(sample_num=self.parameter['样本数'], pallel_num=self.parameter['并行数'])
        model.set_load_shedding_location(self.load_shedding_location)
        model.set_blocking_hvdc_location(self.blocking_hvdc_location)        
        model.load_future_scenario(self.parameter['场景'], self.dyr_file) 
        model.set_system_security_constraint(self.system_security_constraint)

        shedding_percent = model.get_shedding_percent()
        all_sample_feature = []
        k = 0
        while True:
            if k < shedding_percent.shape[0]:
                m = 10
            else:
                m = shedding_percent.shape[0] - k 

            simulation_pars = []
            for j in range(m):
                n = k + j
                par = {'sample_num': n, 'shedding_percent': shedding_percent[n, :]}
                simulation_pars.append(par)                
            
            sample_feature = model.generate_load_shedding_sample_with_parallel_method(simulation_pars)
            self.process_signal.emit(sample_feature)
            all_sample_feature = all_sample_feature + sample_feature 

            k = k + m 
            if k >= shedding_percent.shape[0]:
                break     
        all_sample_feature = np.array(all_sample_feature)
        self.finish_signal.emit([shedding_percent, all_sample_feature])


class UI_SGTL(QDialog):
    def __init__(self):
        super(UI_SGTL, self).__init__()   
        self.setWindowTitle('未来方式下样本生成') 
        self.resize(700, 600)
        self.setWindowIcon(QIcon('.\\data\\safety.png')) 

        self.mainlayout = QGridLayout()
        self.setLayout(self.mainlayout)
        self.parameter_group = QGroupBox('参数设置')
        self.process_group = QGroupBox('生成过程')
        self.mainlayout.addWidget(self.parameter_group, 0, 0)
        self.mainlayout.addWidget(self.process_group, 0, 1)   

        self.init_parameter()
        self.init_process()

        self.sample_thread = SampleGenerationThreadTl()
        self.sample_thread.finish_signal.connect(self.finish_sample_generation)
        self.sample_thread.process_signal.connect(self.refresh_process_information)
        return 

    def init_parameter(self):
        self.parameter_layout = QGridLayout()
        self.parameter_group.setLayout(self.parameter_layout)  

        current_best_scheme_label = QLabel('当前方案')
        self.current_best_scheme_line = QLineEdit()
        self.current_best_scheme_line.setText('.\\data\\多种切负荷方案_example.csv')
        current_best_scheme_button = QPushButton('选择')

        future_scenario_label = QLabel('未来运行方式')
        self.future_scenario_line = QLineEdit()
        self.future_scenario_line.setText('.\\scenarios\\example_scenario_target_up.raw')
        future_scenario_button = QPushButton('选择')
        future_scenario_button.clicked.connect(self.choose_future_scenario)

        sample_num_label = QLabel('样本数')
        self.sample_num_spinbox =  QSpinBox()
        self.sample_num_spinbox.setMinimum(10)
        self.sample_num_spinbox.setMaximum(1000)
        self.sample_num_spinbox.setValue(10)
        self.sample_num_spinbox.setSingleStep(10)    

        self.up_div_rbtn = QRadioButton('上偏移')
        self.down_div_rbtn = QRadioButton('下偏移')
        self.down_div_rbtn.setChecked(True)
        self.parameter_layout.addWidget(self.up_div_rbtn, 3, 0)
        self.parameter_layout.addWidget(self.down_div_rbtn, 3, 1)          

        deviation_label = QLabel('偏移系数')
        self.deviation_spinbox =  QDoubleSpinBox()
        self.deviation_spinbox.setDecimals(4)
        self.deviation_spinbox.setMinimum(0.001)
        self.deviation_spinbox.setMaximum(0.01)
        self.deviation_spinbox.setValue(0.008)
        self.deviation_spinbox.setSingleStep(0.0005) 
        self.parameter_layout.addWidget(deviation_label, 3, 2)
        self.parameter_layout.addWidget(self.deviation_spinbox, 3, 3) 

        pallel_num_label = QLabel('并行数')
        self.pallel_num_spinbox = QSpinBox()
        self.pallel_num_spinbox.setMinimum(1)
        self.pallel_num_spinbox.setMaximum(20)
        self.pallel_num_spinbox.setValue(2)
        self.pallel_num_spinbox.setSingleStep(1)

        self.generation_button = QPushButton('生成')
        self.generation_button.clicked.connect(self.start_sample_generation)
        self.origin_scheme_line = QLineEdit()

        self.parameter_layout.addWidget(current_best_scheme_label, 0, 0)
        self.parameter_layout.addWidget(self.current_best_scheme_line, 0, 1, 1, 2)
        self.parameter_layout.addWidget(current_best_scheme_button, 0, 3)

        self.parameter_layout.addWidget(future_scenario_label, 1, 0)
        self.parameter_layout.addWidget(self.future_scenario_line, 1, 1, 1, 2)
        self.parameter_layout.addWidget(future_scenario_button, 1, 3)

        self.parameter_layout.addWidget(sample_num_label, 2, 0)
        self.parameter_layout.addWidget(self.sample_num_spinbox, 2, 1)
        self.parameter_layout.addWidget(pallel_num_label, 2, 2)
        self.parameter_layout.addWidget(self.pallel_num_spinbox, 2, 3)  

        self.parameter_layout.addWidget(self.generation_button, 5, 3)        
         

    def init_process(self):
        self.process_layout = QGridLayout()
        self.process_group.setLayout(self.process_layout)
        self.process_table = QTableWidget()
        self.process_table.setColumnCount(3)
        self.process_table.setHorizontalHeaderLabels(['编号', '最低频率', '标签'])
        self.process_table.verticalHeader().setVisible(False)  # 隐藏水平表头  
        self.process_layout.addWidget(self.process_table, 0, 0, 1, 4)

        stable_label = QLabel('稳定样本总数')
        self.stable_line = QLineEdit()
        unstable_label = QLabel('失稳样本总数')
        self.unstable_line = QLineEdit()
        self.process_layout.addWidget(stable_label, 1, 0)
        self.process_layout.addWidget(self.stable_line, 1, 1)
        self.process_layout.addWidget(unstable_label, 1, 2)
        self.process_layout.addWidget(self.unstable_line, 1, 3) 

    def start_sample_generation(self):
        self.generation_button.setEnabled(False)
        best_scheme = self.current_best_scheme_line.text()
        sample_num = self.sample_num_spinbox.value()
        pallel_num = self.pallel_num_spinbox.value()
        deviation_num = self.deviation_spinbox.value()

        self.process_table.clearContents()
        self.process_table.setRowCount(sample_num)
        scenario = self.future_scenario_line.text()
        if len(scenario) == 0:
            QMessageBox.warning(self, '警告', '未选择未来场景', QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)
            return 

        if self.up_div_rbtn.isChecked():
            direction = '增加'
        else:
            direction = '减少'
        parameter = {'场景': scenario, '样本数': sample_num, '并行数': pallel_num, '偏移系数':deviation_num,
            '最优方案': best_scheme, '方向': direction}

        self.sample_thread.set_generation_parameter(parameter)
        self.sample_thread.start()  

    def finish_sample_generation(self, singal):
        self.generation_button.setEnabled(True)
        row = self.process_table.rowCount()
        labels = []
        for i in range(row):
            label = self.process_table.item(i, 2).text()
            labels.append(int(label))
        labels = np.array(labels)
        self.stable_line.setText(str(np.sum(labels)))
        self.unstable_line.setText(str(len(labels) - np.sum(labels)))

        QMessageBox.information(self, "提示", "样本生成完毕!", QMessageBox.Yes, QMessageBox.Yes) 

    def choose_future_scenario(self):
        openfile_name = QFileDialog.getOpenFileName(self, '未来运行方式', '' ,'file(*.raw)')
        if len(openfile_name[0]) == 0:
            return
                
        path_name = openfile_name[0]
        if len(path_name) == 0:
            return
        self.future_scenario_line.setText(path_name)          
         

    def refresh_process_information(self, sample_feature):
        for i in range(len(sample_feature)):
            self.process_table.setItem(sample_feature[i][0], 0, QTableWidgetItem(str(sample_feature[i][0]))) 
            frequency = round(sample_feature[i][1], 4)
            self.process_table.setItem(sample_feature[i][0], 1, QTableWidgetItem(str(frequency)))
            self.process_table.setItem(sample_feature[i][0], 2, QTableWidgetItem(str(sample_feature[i][2])))
        self.process_table.verticalScrollBar().setValue(sample_feature[-1][0]) 


class SampleGenerationThreadTl(QThread):#线程类
    finish_signal = pyqtSignal(str)  
    finish_check_signal = pyqtSignal(str)
    process_signal = pyqtSignal(list)
    def __init__(self):
        super(SampleGenerationThreadTl, self).__init__()
        self.load_shedding_location = '.\\data\\loads_shedding.csv'
        self.blocking_hvdc_location = '.\\data\\hvdc_block.csv'
        self.dyr_file = '.\\data\\bench_shandong_change_with_gov.dyr'
        self.system_security_constraint = '.\\data\\safety_constraint.csv'

    def set_generation_parameter(self, parameter):
        self.parameter = parameter

    def run(self): #线程执行函数
        model = SGTL()
        model.set_load_shedding_location(self.load_shedding_location)
        model.set_blocking_hvdc_location(self.blocking_hvdc_location)        
        model.load_future_scenario(self.parameter['场景'], self.dyr_file) 
        model.set_system_security_constraint(self.system_security_constraint)
        model.set_parameter_data('并行数', self.parameter['并行数'])
        model.set_parameter_data('样本数', self.parameter['样本数'])
        model.set_parameter_data('偏移系数', self.parameter['偏移系数'])
        model.set_parameter_data('方向', self.parameter['方向'])
        model.set_best_scheme(self.parameter['最优方案'])

        data = pd.read_csv(self.load_shedding_location, header=0, engine='python')
        loads_shedding = data['负荷'].values.tolist() 
        for i in range(len(loads_shedding)):
            loads_shedding[i] = eval(loads_shedding[i])
        shedding_percent = model.generate_new_scenario_load_shedding_sample()
        with open('.\\data\\x_train_tl.csv', 'w', newline='') as f: 
            csv_write = csv.writer(f)
            csv_write.writerow(loads_shedding)
            csv_write.writerows(shedding_percent)

        all_sample_feature = []
        k = 0
        while True:
            if k < shedding_percent.shape[0]:
                m = 10
            else:
                m = shedding_percent.shape[0] - k 

            simulation_pars = []
            for j in range(m):
                n = k + j
                par = {'sample_num': n, 'scale_percent': shedding_percent[n, :]}
                simulation_pars.append(par)                
            
            sample_feature = model.generate_load_shedding_sample_with_parallel_method(simulation_pars)
            self.process_signal.emit(sample_feature)
            all_sample_feature = all_sample_feature + sample_feature 

            k = k + m 
            if k >= shedding_percent.shape[0]:
                break     
        with open('.\\data\\y_train_tl.csv', 'w', newline='') as f: 
            csv_write = csv.writer(f)
            csv_write.writerow(['编号', '频率', '标签'])
            csv_write.writerows(all_sample_feature) 
        self.finish_signal.emit('完成')
        