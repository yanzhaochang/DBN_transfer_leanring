from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import pandas as pd 
import csv 
import multiprocessing
import sys 
sys.path.append('.\\code')
from DMADE import ADELSM


class UI_AAMO(QDialog):
    def __init__(self):
        super(UI_AAMO, self).__init__()
        self.setWindowTitle('单方案优化')
        self.resize(800, 800)
        self.setWindowIcon(QIcon('.\\data\\safety.png')) 

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
        self.scenario_line.setText('.\\scenarios\\example_scenario_source.raw')

        self.parameter_layout.addWidget(scenario_label, 0, 0)
        self.parameter_layout.addWidget(self.scenario_line, 0, 1, 1, 3)    
        self.parameter_layout.addWidget(scenario_button, 0, 4) 

        ass_model_label = QLabel('辅助模型')
        ass_model_button = QPushButton('选择')
        ass_model_button.clicked.connect(self.choose_ass_model)
        self.ass_model_line = QLineEdit()
        self.ass_model_line.setText('.\\models\\model_source_example.h5')
        self.parameter_layout.addWidget(ass_model_label, 1, 0)
        self.parameter_layout.addWidget(self.ass_model_line, 1, 1, 1, 3) 
        self.parameter_layout.addWidget(ass_model_button, 1, 4)               

        scheme_saving_label = QLabel('优化结果保存')
        self.scheme_saving_line = QLineEdit()
        self.scheme_saving_line.setText('.\\data\\单方案优化结果.csv')
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
        self.optimize_result_table.clearContents()

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

        size = self.size_spinbox.value()
        iter_num = self.iter_num_spinbox.value()
        F = self.F_spinbox.value()
        CR = self.CR_spinbox.value()

        parameter = {'种群数目': size, '进化代数': iter_num, '交叉因子': CR, '变异因子': F,
            '代理模型': ass_model, '场景': raw_file, '保存位置': scheme_saving_file}

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
        self.load_shedding_location = '.\\data\\loads_shedding.csv'
        self.blocking_hvdc_location = '.\\data\\hvdc_block.csv'
        self.dyr_file = '.\\data\\bench_shandong_change_with_gov.dyr'

    def set_generation_parameter(self, parameter):
        self.parameter = parameter

    def run(self): #线程执行函数
        optimizer = ADELSM(size=self.parameter['种群数目'], iter_num=self.parameter['进化代数'])
        optimizer.set_load_shedding_location(self.load_shedding_location)
        optimizer.set_blocking_hvdc_location(self.blocking_hvdc_location)        
        optimizer.load_scenario_data(self.parameter['场景'], self.dyr_file) 

        optimizer.load_frequency_classification_prediction_model(self.parameter['代理模型'])
        optimizer.set_evolution_parameter('种群数目', self.parameter['种群数目'])
        optimizer.set_evolution_parameter('交叉因子', self.parameter['交叉因子'])
        optimizer.set_evolution_parameter('变异因子', self.parameter['变异因子'])

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
        


class UI_AAMOTL(QDialog):
    def __init__(self):
        super(UI_AAMOTL, self).__init__()
        self.setWindowTitle('多方案优化')
        self.resize(800, 800)
        self.setWindowIcon(QIcon('.\\data\\safety.png')) 

        self.mainlayout = QGridLayout()
        self.setLayout(self.mainlayout)
        
        self.parameter_group = QGroupBox('参数设置')
        self.result_group = QGroupBox('优化结果')
        
        self.mainlayout.addWidget(self.parameter_group, 0, 0)
        self.mainlayout.addWidget(self.result_group, 1, 0)
        
        self.init_parameter()
        self.init_optimize_result() 

        self.optimizer = OptimizerTLThread()
        self.optimizer.process_signal.connect(self.display_best_load_shedding_plan)
        self.optimizer.finish_signal.connect(self.save_best_load_shedding_plan)

    def init_parameter(self):
        self.parameter_layout = QGridLayout()
        self.parameter_group.setLayout(self.parameter_layout)

        scenario_label = QLabel('优化场景')
        scenario_button = QPushButton('选择')
        scenario_button.clicked.connect(self.choose_scenario)
        self.scenario_line = QLineEdit()
        self.scenario_line.setText('.\\scenarios\\example_scenario_source.raw')

        self.parameter_layout.addWidget(scenario_label, 0, 0)
        self.parameter_layout.addWidget(self.scenario_line, 0, 1, 1, 3)    
        self.parameter_layout.addWidget(scenario_button, 0, 5) 

        ass_model_label = QLabel('辅助模型')
        ass_model_button = QPushButton('选择')
        ass_model_button.clicked.connect(self.choose_ass_model)
        self.ass_model_line = QLineEdit()
        self.ass_model_line.setText('.\\models\\model_source_example.h5')
        self.parameter_layout.addWidget(ass_model_label, 1, 0)
        self.parameter_layout.addWidget(self.ass_model_line, 1, 1, 1, 3) 
        self.parameter_layout.addWidget(ass_model_button, 1, 5)               

        scheme_saving_label = QLabel('优化结果保存')
        self.scheme_saving_line = QLineEdit()
        self.scheme_saving_line.setText('.\\data\\多种切负荷方案.csv')
        scheme_saving_button = QPushButton('选择')
        scheme_saving_button.clicked.connect(self.save_best_shedding_scheme)
        self.parameter_layout.addWidget(scheme_saving_label, 2, 0)
        self.parameter_layout.addWidget(self.scheme_saving_line, 2, 1, 1, 3) 
        self.parameter_layout.addWidget(scheme_saving_button, 2, 5) 

        scheme_num_label = QLabel('方案数目')
        self.scheme_num_spinbox = QSpinBox()
        self.scheme_num_spinbox.setMinimum(20)
        self.scheme_num_spinbox.setMaximum(500)
        self.scheme_num_spinbox.setValue(100)
        self.scheme_num_spinbox.setSingleStep(10)        
        self.parameter_layout.addWidget(scheme_num_label, 3, 0)
        self.parameter_layout.addWidget(self.scheme_num_spinbox, 3, 1)

        parallel_num_label = QLabel('并行数')
        self.parallel_num_spinbox = QSpinBox()
        self.parallel_num_spinbox.setMinimum(1)
        self.parallel_num_spinbox.setMaximum(20)
        self.parallel_num_spinbox.setValue(5)
        self.parameter_layout.addWidget(parallel_num_label, 4, 0)
        self.parameter_layout.addWidget(self.parallel_num_spinbox, 4, 1)

        size_label = QLabel('种群数')
        self.size_spinbox = QSpinBox()
        self.size_spinbox.setMinimum(20)
        self.size_spinbox.setMaximum(200)
        self.size_spinbox.setValue(50)
        self.size_spinbox.setSingleStep(10)        
        self.parameter_layout.addWidget(size_label, 5, 0)
        self.parameter_layout.addWidget(self.size_spinbox, 5, 1)
        
        F_label = QLabel('    交叉因子')
        self.F_spinbox = QDoubleSpinBox()
        self.F_spinbox.setMinimum(0.1)
        self.F_spinbox.setMaximum(1.5)
        self.F_spinbox.setValue(0.5)
        self.F_spinbox.setSingleStep(0.05)  
        self.F_spinbox.setDecimals(2)        
        self.parameter_layout.addWidget(F_label, 3, 2)
        self.parameter_layout.addWidget(self.F_spinbox, 3, 3)
        
        CR_label = QLabel('    变异因子')
        self.CR_spinbox = QDoubleSpinBox()
        self.CR_spinbox.setMinimum(0.1)
        self.CR_spinbox.setMaximum(1.5)
        self.CR_spinbox.setValue(0.5)
        self.CR_spinbox.setSingleStep(0.05)   
        self.CR_spinbox.setDecimals(2)       
        self.parameter_layout.addWidget(CR_label, 4, 2)
        self.parameter_layout.addWidget(self.CR_spinbox, 4, 3)        
        
        iter_num_label = QLabel('    进化代数')
        self.iter_num_spinbox = QSpinBox()
        self.iter_num_spinbox.setMinimum(10)
        self.iter_num_spinbox.setMaximum(2000)
        self.iter_num_spinbox.setValue(200)
        self.iter_num_spinbox.setSingleStep(5)          
        self.parameter_layout.addWidget(iter_num_label, 5, 2)
        self.parameter_layout.addWidget(self.iter_num_spinbox, 5, 3) 
        
        self.button_run = QPushButton('计算')
        self.parameter_layout.addWidget(self.button_run, 6, 5)
        self.button_run.clicked.connect(self.start_optimizer)
         
    
    def init_optimize_result(self):
        self.optimize_result_layout = QGridLayout()
        self.result_group.setLayout(self.optimize_result_layout)

        data = pd.read_csv('.\data\\loads_shedding.csv', header=0, engine='python')
        loads_shedding = data['负荷'].values.tolist() 

        self.optimize_result_table = QTableWidget()
        self.optimize_result_table.setColumnCount(len(loads_shedding)+1)   
        self.optimize_result_table.setHorizontalHeaderLabels(loads_shedding+['总量'])
        self.optimize_result_layout.addWidget(self.optimize_result_table, 0, 0)


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

    def save_best_shedding_scheme(self):
        openfile_name = QFileDialog.getSaveFileName(self, '保存方案', '' ,'file(*.csv)')
        if len(openfile_name[0]) == 0:
            return
        path_name = openfile_name[0]
        self.scheme_saving_line.setText(path_name)    


    def start_optimizer(self):
        self.button_run.setEnabled(False)
        self.optimize_result_table.clearContents()
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

        size = self.size_spinbox.value()
        iter_num = self.iter_num_spinbox.value()
        F = self.F_spinbox.value()
        CR = self.CR_spinbox.value()
        parallel_num = self.parallel_num_spinbox.value()
        scheme_num = self.scheme_num_spinbox.value()


        parameter = {'种群数目': size, '进化代数': iter_num, '交叉因子': CR, '变异因子': F,
            '代理模型': ass_model, '场景': raw_file, '保存位置': scheme_saving_file,
            '并行数': parallel_num, '方案数': scheme_num}

        self.optimizer.set_generation_parameter(parameter)
        self.optimizer.start()        

    def display_best_load_shedding_plan(self, result):
        row = self.optimize_result_table.rowCount()
        self.optimize_result_table.setRowCount(row+len(result))
        column = self.optimize_result_table.columnCount()
        for i in range(len(result)):
            for j in range(len(result[0])):
                value = round(result[i][j], 4)
                self.optimize_result_table.setItem(row + i, j, QTableWidgetItem(str(value)))
        self.optimize_result_table.scrollToBottom()

    def save_best_load_shedding_plan(self, information):
        self.button_run.setEnabled(True)
        QMessageBox.information(self, "提示",  self.tr("方案生成完成"))
        row = self.optimize_result_table.rowCount()
        column = self.optimize_result_table.columnCount()
        scheme = []
        for i in range(row):
            single_scheme = []
            for j in range(column-1):
                value = self.optimize_result_table.item(i, j).text()
                value = float(value)
                single_scheme.append(value)
            scheme.append(single_scheme)
        
        data = pd.read_csv('.\\data\\loads_shedding.csv', header=0, engine='python')
        loads_shedding = data['负荷'].values.tolist() 
        for i in range(len(loads_shedding)):
            loads_shedding[i] = eval(loads_shedding[i])

        with open(self.scheme_saving_line.text(), 'w', newline='') as f: 
            csv_write = csv.writer(f)
            csv_write.writerow(loads_shedding)
            csv_write.writerows(scheme)       
        return 


class OptimizerTLThread(QThread):  
    process_signal = pyqtSignal(list)
    finish_signal = pyqtSignal(str)
    def __init__(self):
        super(OptimizerTLThread, self).__init__()
        self.dyr_file = '.\\data\\bench_shandong_change_with_gov.dyr'

    def set_generation_parameter(self, parameter):
        self.parameter = parameter

    def run(self): #线程执行函数
        k = 0
        while True:
            if k < self.parameter['方案数']:
                m = 10
            else:
                m = self.parameter['方案数'] - k 
            simulation_pars = []
            for j in range(m):
                n = k + j
                d1 = {'sample_num': n}
                par = dict(d1, **self.parameter)
                simulation_pars.append(par)

            best_inds = optimize_scenario_loads_shedding_with_parallel_method(simulation_pars)
            self.process_signal.emit(best_inds)
            k = k + m 
            if k >= self.parameter['方案数']:
                break 
        self.finish_signal.emit('完成')
        return 

def optimize_scenario_loads_shedding_with_parallel_method(simulation_pars): 
    p = multiprocessing.Pool(processes=simulation_pars[0]['并行数'])  
    best_inds = p.map(optimize_scenario_loads_shedding, simulation_pars)
    p.close()
    p.join()
    return  best_inds   

def optimize_scenario_loads_shedding(simulation_pars):
    from DMADE import ADELSMTL
    sample_num = simulation_pars['sample_num']

    optimizer = ADELSMTL(size=simulation_pars['种群数目'], iter_num=simulation_pars['进化代数'],)
    optimizer.load_frequency_classification_prediction_model(simulation_pars['代理模型'])
    optimizer.set_load_shedding_location('.\\data\\loads_shedding.csv')
    dyr_file = '.\\data\\bench_shandong_change_with_gov.dyr'
    optimizer.load_scenario_data(simulation_pars['场景'], dyr_file)
    optimizer.initialize_population()
    optimizer.operate_evolution()
    best_ind, min_shedding_power = optimizer.get_best_individual()
    del optimizer       
    return best_ind.tolist() + [min_shedding_power]