from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import pandas as pd 
import sys 
sys.path.append('.\\code')
from DMADE import ADELSM

class UI_AAMO(QDialog):
    def __init__(self):
        super(UI_AAMO, self).__init__()
        self.setWindowTitle('代理辅助模型优化计算')
        self.resize(800, 800)
        self.setWindowIcon(QIcon('.\\logo\\安全.png')) 

        self.mainlayout = QGridLayout()
        self.setLayout(self.mainlayout)
        
        self.parameter_group = QGroupBox('参数设置')
        self.result_group = QGroupBox('优化结果')
        self.process_group = QGroupBox('优化过程')
        
        self.mainlayout.addWidget(self.parameter_group, 0, 0)
        self.mainlayout.addWidget(self.process_group, 0, 1)
        self.mainlayout.addWidget(self.result_group, 1, 0, 1, 2)
        
        self.init_parameter()
        self.init_process()
        self.init_optimize_result()

        self.optimizer = OptimizerThread()
        self.optimizer.process_signal.connect(self.refresh_evolution_process)
        self.optimizer.time_simulation_signal.connect(self.operate_time_domain_simulation)
        self.optimizer.finish_signal.connect(self.display_best_load_shedding_plan)

    def init_parameter(self):
        self.parameter_layout = QGridLayout()
        self.parameter_group.setLayout(self.parameter_layout)

        scenario_label = QLabel('优化场景')
        scenario_button = QPushButton('选择')
        scenario_button.clicked.connect(self.choose_scenario)
        self.scenario_line = QLineEdit()
        self.scenario_line.setText('.\\运行方式\\example_scenario_source.raw')

        self.parameter_layout.addWidget(scenario_label, 0, 0)
        self.parameter_layout.addWidget(self.scenario_line, 0, 1, 1, 3)    
        self.parameter_layout.addWidget(scenario_button, 0, 4) 

        ass_model_label = QLabel('辅助模型')
        ass_model_button = QPushButton('选择')
        ass_model_button.clicked.connect(self.choose_ass_model)
        self.ass_model_line = QLineEdit()
        self.ass_model_line.setText('.\\代理模型\\model_source_example.h5')
        self.parameter_layout.addWidget(ass_model_label, 1, 0)
        self.parameter_layout.addWidget(self.ass_model_line, 1, 1, 1, 3) 
        self.parameter_layout.addWidget(ass_model_button, 1, 4)       

        origin_scheme_label = QLabel('历史方案')
        self.origin_scheme_line = QLineEdit()
        origin_scheme_button = QPushButton('选择')
        origin_scheme_button.clicked.connect(self.choose_origin_scheme)
        self.parameter_layout.addWidget(origin_scheme_label, 2, 0)
        self.parameter_layout.addWidget(self.origin_scheme_line, 2, 1, 1, 3) 
        self.parameter_layout.addWidget(origin_scheme_button, 2, 4)         

        scheme_saving_label = QLabel('优化结果保存')
        self.scheme_saving_line = QLineEdit()
        self.scheme_saving_line.setText('.\\优化结果\\最优切负荷方案.csv')
        scheme_saving_button = QPushButton('选择')
        scheme_saving_button.clicked.connect(self.save_best_shedding_scheme)
        self.parameter_layout.addWidget(scheme_saving_label, 3, 0)
        self.parameter_layout.addWidget(self.scheme_saving_line, 3, 1, 1, 3) 
        self.parameter_layout.addWidget(scheme_saving_button, 3, 4) 

        size_label = QLabel('种群数')
        self.size_spinbox = QSpinBox()
        self.size_spinbox.setMinimum(20)
        self.size_spinbox.setMaximum(200)
        self.size_spinbox.setValue(50)
        self.size_spinbox.setSingleStep(10)        
        self.parameter_layout.addWidget(size_label, 6, 0)
        self.parameter_layout.addWidget(self.size_spinbox, 6, 1)
        
        F_label = QLabel('交叉因子')
        self.F_spinbox = QDoubleSpinBox()
        self.F_spinbox.setMinimum(0.1)
        self.F_spinbox.setMaximum(1.5)
        self.F_spinbox.setValue(0.5)
        self.F_spinbox.setSingleStep(0.05)  
        self.F_spinbox.setDecimals(2)        
        self.parameter_layout.addWidget(F_label, 7, 0)
        self.parameter_layout.addWidget(self.F_spinbox, 7, 1)
        
        CR_label = QLabel('变异因子')
        self.CR_spinbox = QDoubleSpinBox()
        self.CR_spinbox.setMinimum(0.1)
        self.CR_spinbox.setMaximum(1.5)
        self.CR_spinbox.setValue(0.5)
        self.CR_spinbox.setSingleStep(0.05)   
        self.CR_spinbox.setDecimals(2)       
        self.parameter_layout.addWidget(CR_label, 8, 0)
        self.parameter_layout.addWidget(self.CR_spinbox, 8, 1)        
        
        iter_num_label = QLabel('进化代数')
        self.iter_num_spinbox = QSpinBox()
        self.iter_num_spinbox.setMinimum(10)
        self.iter_num_spinbox.setMaximum(2000)
        self.iter_num_spinbox.setValue(200)
        self.iter_num_spinbox.setSingleStep(5)          
        self.parameter_layout.addWidget(iter_num_label, 9, 0)
        self.parameter_layout.addWidget(self.iter_num_spinbox, 9, 1) 
        
        self.button_run = QPushButton('计算')
        self.parameter_layout.addWidget(self.button_run, 10, 4)
        self.button_run.clicked.connect(self.start_optimizer)
    
    def init_process(self):
        self.process_layout = QGridLayout()
        self.process_group.setLayout(self.process_layout)
        self.process_textedit = QTextEdit()
        self.process_textedit.setPlainText('差分进化算法代理辅助模型优化')
        self.process_layout.addWidget(self.process_textedit, 0, 0)
         
    
    def init_optimize_result(self):
        self.optimize_result_layout = QGridLayout()
        self.result_group.setLayout(self.optimize_result_layout)
        
        self.optimize_result_table = QTableWidget()
        self.optimize_result_table.setColumnCount(5)
        self.optimize_result_table.setRowCount(10)   
        self.optimize_result_table.setHorizontalHeaderLabels(['切负荷点', '母线名', '负荷量/MW', '切除比例/%', '切除量/MW'])
        self.optimize_result_layout.addWidget(self.optimize_result_table, 0, 0)
        
        self.optimize_result_group = QGroupBox()
        self.optimize_result_group.setFixedWidth(200)
        self.optimize_result_widget_layout = QGridLayout()
        self.load_shedding_all_power_label = QLabel('切负荷总量')
        self.load_shedding_all_power_line = QLineEdit('0.0  MW')
        self.load_shedding_all_power_line.setFixedWidth(100)

        self.time_domain_simulation_label = QLabel('时域仿真校验')
        self.min_frequency_label = QLabel('最低频率')
        self.min_frequency_result_line = QLineEdit('0.0 Hz')
        self.min_frequency_result_line.setFixedWidth(100)
        self.optimize_result_widget_layout.addWidget(self.load_shedding_all_power_label, 0, 0)
        self.optimize_result_widget_layout.addWidget(self.load_shedding_all_power_line, 0, 1)
        self.optimize_result_widget_layout.addWidget(self.time_domain_simulation_label, 1, 0)
        self.optimize_result_widget_layout.addWidget(self.min_frequency_label, 2, 0)
        self.optimize_result_widget_layout.addWidget(self.min_frequency_result_line, 2, 1)
        self.optimize_result_group.setLayout(self.optimize_result_widget_layout)
        self.optimize_result_layout.addWidget(self.optimize_result_group, 0, 1) 
    
    def start_optimizer(self):
        self.button_run.setEnabled(False)

        ass_model = self.ass_model_line.text()
        if len(ass_model) == 0:
            QMessageBox.warning(self,'警告','未选择代理辅助模型', QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)
            return 
        raw_file = self.scenario_line.text()
        if len(raw_file) == 0:
            QMessageBox.warning(self,'警告','未选择优化场景', QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)
            return  
        
        scheme_saving_file = self.scheme_saving_line.text()
        if len(scheme_saving_file) == 0:
            QMessageBox.warning(self,'警告','未选择优化结果保存位置', QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)
            return

        origin_scheme = self.origin_scheme_line.text()
        size = self.size_spinbox.value()
        iter_num = self.iter_num_spinbox.value()
        F = self.F_spinbox.value()
        CR = self.CR_spinbox.value()

        parameter = {'种群数目': size, '进化代数': iter_num, '交叉因子': CR, '变异因子': F,
            '代理模型': ass_model, '场景': raw_file, '历史方案': origin_scheme, '保存位置': scheme_saving_file}

        self.optimizer.set_generation_parameter(parameter)
        self.optimizer.start()

    def choose_ass_model(self):
        openfile_name = QFileDialog.getOpenFileName(self, '代理模型', '' ,'file(*.h5)')
        if len(openfile_name[0]) == 0:
            return       
        path_name = openfile_name[0]
        self.ass_model_line.setText(path_name)         
        return 

    def choose_scenario(self):
        openfile_name = QFileDialog.getOpenFileName(self, '场景', '' ,'file(*.raw)')
        if len(openfile_name[0]) == 0:
            return               
        path_name = openfile_name[0]
        self.scenario_line.setText(path_name)  

    def choose_origin_scheme(self):
        openfile_name = QFileDialog.getOpenFileName(self, '历史方案', '' ,'file(*.csv)')
        if len(openfile_name[0]) == 0:
            return
        path_name = openfile_name[0]
        self.origin_scheme_line.setText(path_name)          

    def save_best_shedding_scheme(self):
        openfile_name = QFileDialog.getSaveFileName(self, '保存方案', '' ,'file(*.csv)')
        if len(openfile_name[0]) == 0:
            return
        path_name = openfile_name[0]
        self.scheme_saving_line.setText(path_name)          

    def display_best_load_shedding_plan(self, result):
        best_individual, loads_p, loads_shedding, loads_name = result[0], result[1], result[2], result[3]
        
        shedding_power = best_individual * loads_p
        power = round(sum(shedding_power), 4)
        self.load_shedding_all_power_line.setText(str(power) + '    MW')
        self.optimize_result_table.setRowCount(len(loads_shedding))   
        for i in range(len(loads_shedding)):
            self.optimize_result_table.setItem(i, 0, QTableWidgetItem(str(loads_shedding[i])))
            self.optimize_result_table.setItem(i, 1, QTableWidgetItem(loads_name[i]))

            power = round(loads_p[i], 4)
            self.optimize_result_table.setItem(i, 2, QTableWidgetItem(str(power)))
            percent = round(best_individual[i]*100, 4)
            self.optimize_result_table.setItem(i, 3, QTableWidgetItem(str(percent)))
            power = round(shedding_power[i], 4)
            self.optimize_result_table.setItem(i, 4, QTableWidgetItem(str(power)))
        return 

    def refresh_evolution_process(self, information):
        self.process_textedit.append(information)
        self.process_textedit.verticalScrollBar().setValue(self.process_textedit.verticalScrollBar().maximum())
        QApplication.processEvents()

    def operate_time_domain_simulation(self, information):
        self.min_frequency_result_line.setText(information)
        self.button_run.setEnabled(True)
        self.process_textedit.append('优化计算完毕')
        self.process_textedit.verticalScrollBar().setValue(self.process_textedit.verticalScrollBar().maximum())


class OptimizerThread(QThread):  
    process_signal = pyqtSignal(str)
    finish_signal = pyqtSignal(list)
    time_simulation_signal = pyqtSignal(str)
    def __init__(self):
        super(OptimizerThread, self).__init__()
        self.load_shedding_location = '.\\参数设置\\切负荷站.csv'
        self.blocking_hvdc_location = '.\\参数设置\\闭锁直流.csv'
        self.dyr_file = '.\\参数设置\\bench_shandong_change_with_gov.dyr'

    def set_generation_parameter(self, parameter):
        self.parameter = parameter

    def run(self): #线程执行函数
        optimizer = ADELSM()
        optimizer.set_load_shedding_location(self.load_shedding_location)
        optimizer.set_blocking_hvdc_location(self.blocking_hvdc_location)        
        optimizer.load_scenario_data(self.parameter['场景'], self.dyr_file) 

        optimizer.load_frequency_classification_prediction_model(self.parameter['代理模型'])
        optimizer.set_evolution_parameter('种群数目', self.parameter['种群数目'])
        optimizer.set_evolution_parameter('交叉因子', self.parameter['交叉因子'])
        optimizer.set_evolution_parameter('变异因子', self.parameter['变异因子'])
        optimizer.set_evolution_parameter('历史方案', self.parameter['历史方案'])

        optimizer.initialize_population()
        k = 0
        while True:
            mean_fitness, max_fitness, min_fitness = optimizer.operate_evolution_one_enpouch()
     
            self.process_signal.emit('第{}代-平均值{}-最大值{}-最小值{}'.format(k, mean_fitness, max_fitness, min_fitness)) 
            k = k + 1
            if k > self.parameter['进化代数']:
                break
        
        best_ind, loads_p, loads_shedding, loads_name = optimizer.save_best_individual(self.parameter['保存位置'])
        
        self.finish_signal.emit([best_ind, loads_p, loads_shedding, loads_name])
        self.process_signal.emit('时域仿真校验中')
        min_frequency = optimizer.check_evolution_result(best_ind)
        self.time_simulation_signal.emit(str(round(min_frequency, 6)) + 'Hz')
        
