from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

import csv 
import numpy as np 
import sys 
sys.path.append('.\\code')
from SampleGeneration import SGB, SGTL 



class UI_SGBM(QDialog):
    message_singal = pyqtSignal(str)
    def __init__(self):
        super(UI_SGBM, self).__init__()
        self.setWindowTitle('样本生成')
        self.resize(700, 500)
        self.setWindowIcon(QIcon('.\\logo\\安全.png'))

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
        self.scenario_line.setText('.\\运行方式\\example_scenario_source.raw')
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
        sample_num = self.sample_num_spinbox.value()
        pallel_num = self.pallel_num_spinbox.value()
        test_data_percent = self.test_data_percent_spinbox.value()

        scenario = self.scenario_line.text()
        if len(scenario) == 0:
            dialog = QMessageBox(QMessageBox.Warning, '警告', '未导入电网目标运行方式!!!!')
            dialog.exec_()
            return

        self.sample_thread.set_generation_parameter({'场景': scenario, '样本数': sample_num, '并行数': pallel_num, '测试集比例': test_data_percent})    
        self.button_generation.setEnabled(False)
        
        self.process_table.setRowCount(sample_num)
        self.sample_thread.start()
        
    def finish_generation_process(self, str_singal):
        self.button_generation.setEnabled(True)

        row = self.process_table.rowCount()
        labels = []
        for i in range(row):
            label = self.process_table.item(i, 2).text()
            labels.append(int(label))
        labels = np.array(labels)
        self.stable_line.setText(str(np.sum(labels)))
        self.unstable_line.setText(str(len(labels) - np.sum(labels)))

        QMessageBox.information(self, "温馨提示", "样本生成完毕!", QMessageBox.Yes, QMessageBox.Yes)
        
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
        self.process_table.verticalScrollBar().setValue(sample_feature[-1][0])         


class SampleGenerationThread(QThread):
    finish_signal = pyqtSignal(str)  
    process_signal = pyqtSignal(list)

    def __init__(self):
        super(SampleGenerationThread, self).__init__()
        self.load_shedding_location = '.\\参数设置\\切负荷站.csv'
        self.blocking_hvdc_location = '.\\参数设置\\闭锁直流.csv'
        self.dyr_file = '.\\参数设置\\bench_shandong_change_with_gov.dyr'
        self.system_security_constraint = '.\\参数设置\\系统安全约束.csv'
         

    def set_generation_parameter(self, parameter):
        self.parameter = parameter
         

    def run(self): #线程执行函数
        model = SGB(sample_num=self.parameter['样本数'], pallel_num=self.parameter['并行数'], test_size=self.parameter['测试集比例'])
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
                par = {'sample_num': n, 'scale_percent': shedding_percent[n, :]}
                simulation_pars.append(par)                
            
            sample_feature = model.generate_load_shedding_sample_with_parallel_method(simulation_pars)
            self.process_signal.emit(sample_feature)
            all_sample_feature = all_sample_feature + sample_feature 

            k = k + m 
            if k >= shedding_percent.shape[0]:
                break     
        all_sample_feature = np.array(all_sample_feature)

        model.spilt_sample(shedding_percent, all_sample_feature)
        self.finish_signal.emit('样本生成完成')


class UI_SGTL(QDialog):
    def __init__(self):
        super(UI_SGTL, self).__init__()   
        self.setWindowTitle('未来方式下样本生成') 
        self.resize(700, 600)
        self.setWindowIcon(QIcon('.\\logo\\安全.png')) 

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
        self.sample_thread.finish_check_signal.connect(self.finish_check_best_sheme)
        self.sample_thread.process_signal.connect(self.refresh_process_information)
        return 

    def init_parameter(self):
        self.parameter_layout = QGridLayout()
        self.parameter_group.setLayout(self.parameter_layout)  

        current_best_scheme_label = QLabel('当前方案')
        self.current_best_scheme_line = QLineEdit()
        self.current_best_scheme_line.setText('.\\优化结果\\最优切负荷方案.csv')
        current_best_scheme_button = QPushButton('选择')

        future_scenario_label = QLabel('未来运行方式')
        self.future_scenario_line = QLineEdit()
        future_scenario_button = QPushButton('选择')
        future_scenario_button.clicked.connect(self.choose_future_scenario)

        sample_num_label = QLabel('样本数')
        self.sample_num_spinbox =  QSpinBox()
        self.sample_num_spinbox.setMinimum(10)
        self.sample_num_spinbox.setMaximum(1000)
        self.sample_num_spinbox.setValue(10)
        self.sample_num_spinbox.setSingleStep(10)    

        sample_scale_label = QLabel('样本分布系数')
        self.sample_scale_spinbox =  QDoubleSpinBox()
        self.sample_scale_spinbox.setDecimals(3)
        self.sample_scale_spinbox.setMinimum(0.001)
        self.sample_scale_spinbox.setMaximum(0.05)
        self.sample_scale_spinbox.setValue(0.01)
        self.sample_scale_spinbox.setSingleStep(0.002) 

        deviation_label = QLabel('偏移系数')
        self.deviation_spinbox =  QDoubleSpinBox()
        self.deviation_spinbox.setMinimum(0.1)
        self.deviation_spinbox.setMaximum(0.9)
        self.deviation_spinbox.setValue(0.5)
        self.deviation_spinbox.setSingleStep(0.05) 

        pallel_num_label = QLabel('并行数')
        self.pallel_num_spinbox = QSpinBox()
        self.pallel_num_spinbox.setMinimum(1)
        self.pallel_num_spinbox.setMaximum(20)
        self.pallel_num_spinbox.setValue(2)
        self.pallel_num_spinbox.setSingleStep(1)

        self.check_origin_scheme_button = QPushButton('校验原方案')
        self.check_origin_scheme_button.clicked.connect(self.check_origin_scheme)
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

        self.parameter_layout.addWidget(sample_scale_label, 3, 0)
        self.parameter_layout.addWidget(self.sample_scale_spinbox, 3, 1) 
        self.parameter_layout.addWidget(deviation_label, 3, 2)
        self.parameter_layout.addWidget(self.deviation_spinbox, 3, 3) 

        self.parameter_layout.addWidget(self.check_origin_scheme_button, 4, 0)
        self.parameter_layout.addWidget(self.origin_scheme_line, 4, 1, 1, 3)
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
        best_scheme = self.current_best_scheme_line.text()
        sample_scale = self.sample_scale_spinbox.value()
        sample_num = self.sample_num_spinbox.value()
        pallel_num = self.pallel_num_spinbox.value()
        deviation_num = self.deviation_spinbox.value()

        self.process_table.setRowCount(sample_num)
        scenario = self.future_scenario_line.text()
        if len(scenario) == 0:
            QMessageBox.warning(self, '警告', '未选择未来场景', QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)
            return 
        parameter = {'场景': scenario, '样本数': sample_num, '并行数': pallel_num, '分布系数': sample_scale, '偏移系数':deviation_num,
            '最优方案': best_scheme}

        self.sample_thread.set_generation_parameter(parameter)
        self.sample_thread.check_signal = False 
        self.generation_button.setEnabled(False)
        self.sample_thread.start()
        return  

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

        QMessageBox.information(self, "温馨提示", "样本生成完毕!", QMessageBox.Yes, QMessageBox.Yes) 

    def check_origin_scheme(self):
        self.check_origin_scheme_button.setEnabled(False)
        best_scheme = self.current_best_scheme_line.text()
        sample_scale = self.sample_scale_spinbox.value()
        sample_num = self.sample_num_spinbox.value()
        pallel_num = self.pallel_num_spinbox.value()
        deviation_num = self.deviation_spinbox.value()

        scenario = self.future_scenario_line.text()
        if len(scenario) == 0:
            QMessageBox.warning(self, '警告', '未选择未来场景', QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)
            return 
        parameter = {'场景': scenario, '样本数': sample_num, '并行数': pallel_num, '分布系数': sample_scale, '偏移系数':deviation_num,
            '最优方案': best_scheme}

        self.sample_thread.set_generation_parameter(parameter)
        self.sample_thread.check_signal = True 
        self.sample_thread.start()
        return 

    def choose_future_scenario(self):
        openfile_name = QFileDialog.getOpenFileName(self, '未来运行方式', '' ,'file(*.raw)')
        if len(openfile_name[0]) == 0:
            return
                
        path_name = openfile_name[0]
        if len(path_name) == 0:
            return
        self.future_scenario_line.setText(path_name)          
        return 

    def finish_check_best_sheme(self, min_frequency):
        self.check_origin_scheme_button.setEnabled(True)
        self.origin_scheme_line.setText('最低频率: ' + min_frequency + 'Hz')

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
        self.load_shedding_location = '.\\参数设置\\切负荷站.csv'
        self.blocking_hvdc_location = '.\\参数设置\\闭锁直流.csv'
        self.dyr_file = '.\\参数设置\\bench_shandong_change_with_gov.dyr'
        self.system_security_constraint = '.\\参数设置\\系统安全约束.csv'
        
        self.check_signal = False 

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
        model.set_parameter_data('分布系数', self.parameter['分布系数'])
        model.set_parameter_data('偏移系数', self.parameter['偏移系数'])
        model.set_best_scheme(self.parameter['最优方案'])

        if self.check_signal is True:
            min_frequency = model.check_origin_scheme()
            self.finish_check_signal.emit(str(min_frequency))
            return

        shedding_percent = model.generate_new_scenario_load_shedding_sample()

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
        model.save_sample_feature(all_sample_feature)
        self.finish_signal.emit('训练完成')
        