# 主界面
import sys
import time
from stepspy import STEPS 
import pandas as pd 
import csv 

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from UI_SampleGeneration import UI_SGBM, UI_SGTL
from UI_ModelTraining import UI_DBNBM, UI_DBNTL
from UI_Optimizer import UI_AAMO


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.main_table = QTabWidget()
        self.setCentralWidget(self.main_table)
        self.setWindowTitle('电力系统紧急切负荷优化软件')
        self.setWindowIcon(QIcon('.\\logo\\安全.png'))
        self.main_table.setStyleSheet('background-color: grey')
        self.main_table.setTabPosition(QTabWidget.South)
        
        self.simulator = STEPS(is_default=False, log_file='.\\log\\log.txt')
        
        self.init_window_menu()
        self.window_output()

    def init_window_menu(self):
        bar = self.menuBar()
        
        basic_parameter = bar.addMenu('基础设置')
        basic_parameter.addAction('电网模型') 
        basic_parameter.addAction('导入切负荷站设置')
        basic_parameter.addAction('保存切负荷站设置')
        basic_parameter.addAction('导入直流闭锁设置')
        basic_parameter.addAction('保存直流闭锁设置')
        basic_parameter.addAction('安全约束设置')
        basic_parameter.triggered[QAction].connect(self.set_basic_parameter)
        
        basic_model_training = bar.addMenu('辅助模型训练')
        basic_model_training.addAction('样本生成')
        basic_model_training.addAction('模型训练')
        basic_model_training.triggered[QAction].connect(self.train_basic_model)

        model_update = bar.addMenu('模型更新')
        model_update.addAction('样本生成')
        model_update.addAction('模型微调')
        model_update.triggered[QAction].connect(self.update_model)
    
        optimizer = bar.addMenu('代理辅助优化')
        optimizer.addAction('启动')
        optimizer.triggered[QAction].connect(self.start_scenraio_optimizer)

        setting = bar.addMenu('帮助')
        setting.addAction('版权信息')
        setting.addAction('优化计算说明')

         
    def window_output(self):
        items = QDockWidget('输出', self)
        self.text_output = QTextBrowser()
        items.setWidget(self.text_output)
        items.setFloating(False)
        items.setFeatures(items.NoDockWidgetFeatures)
        self.addDockWidget(Qt.BottomDockWidgetArea, items)
        
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        return 
      

    def set_basic_parameter(self, q):
        if q.text() == '电网模型':
            openfile_name = QFileDialog.getOpenFileName(self, '选择电网模型', '' ,'file(*.raw)')
            if len(openfile_name[0]) == 0:
                return
            path_raw_file = openfile_name[0]
            if len(path_raw_file) == 0:
                return 
            self.path_raw_file = path_raw_file
            self.simulator.load_powerflow_data(path_raw_file, 'PSS/E')
            self.text_output.append('加载电网模型： ' + path_raw_file)
            self.display_network_data() 
        
        elif q.text() == '导入切负荷站设置':
            try:
                object.__getattribute__(self, 'scenario_table')
            except:
                dialog = QMessageBox(QMessageBox.Warning, '警告', '未导入电网运行文件！！！')
                dialog.exec_()
                return 

            openfile_name = QFileDialog.getOpenFileName(self, '导入切负荷站设置', '' ,'file(*.csv)')
            if len(openfile_name[0]) == 0:
                return
            path_name = openfile_name[0]
            if len(path_name) == 0:
                return
            self.scenario_table.set_loads_shedding_location(path_name) 
            self.text_output.append('导入切负荷站设置： ' + path_name)

        elif q.text() == '保存切负荷站设置': 
            try:
                object.__getattribute__(self, 'scenario_table')
            except:
                dialog = QMessageBox(QMessageBox.Warning, '警告', '未导入电网运行文件！！！')
                dialog.exec_()
                return 

            self.scenario_table.save_loads_shedding_location('.\\参数设置\\切负荷站.csv') 
            self.text_output.append('保存切负荷站设置： ' + '.\\参数设置\\切负荷站.csv')
            return 

        elif q.text() == '导入直流闭锁设置':
            try:
                object.__getattribute__(self, 'scenario_table')
            except:
                dialog = QMessageBox(QMessageBox.Warning, '警告', '未导入电网运行文件！！！')
                dialog.exec_()
                return 

            openfile_name = QFileDialog.getOpenFileName(self, '导入直流闭锁设置', '' ,'file(*.csv)')
            if len(openfile_name[0]) == 0:
                return
            path_name = openfile_name[0]
            if len(path_name) == 0:
                return 
            self.scenario_table.set_block_hvdc_location(path_name)   
            self.text_output.append('导入闭锁直流设置： ' + path_name)

        elif q.text() == '保存直流闭锁设置':
            try:
                object.__getattribute__(self, 'scenario_table')
            except:
                dialog = QMessageBox(QMessageBox.Warning, '警告', '未导入电网运行文件！！！')
                dialog.exec_()
                return 
            self.scenario_table.save_hvdc_blocking_location('.\\参数设置\\闭锁直流.csv')  
            self.text_output.append('保存闭锁直流设置： ' + '.\\参数设置\\闭锁直流.csv')    

        elif q.text() == '安全约束设置':
            try:
                object.__getattribute__(self, 'scenario_table')
            except:
                dialog = QMessageBox(QMessageBox.Warning, '警告', '未导入电网运行文件！！！')
                dialog.exec_()
                return 

            dialog = SecurityConstraint(self.text_output)
            dialog.exec_()            
        else:
            pass 
        return 
    
    def train_basic_model(self, q):
        if q.text() == '样本生成':
            dialog = UI_SGBM()
            dialog.exec_() 

        elif q.text() == '模型训练':    
            dialog = UI_DBNBM()
            dialog.exec_() 

        else:
            return 

    def update_model(self, q):
        if q.text() == '样本生成':
            dialog = UI_SGTL()
            dialog.exec_()
        elif q.text() == '模型微调':
            dialog = UI_DBNTL()
            dialog.exec_()                 
        else:
            return  
   
    def start_scenraio_optimizer(self, q):
        if q.text() == '启动':
            dialog = UI_AAMO()
            dialog.exec_()
        else:
            return             



    def display_network_data(self):
        self.main_table.setStyleSheet('background-color: white')
        self.scenario_table = Table_Show(self.main_table, self.simulator, self.text_output)
        self.scenario_table.display_bus_table()
        self.scenario_table.display_generator_table()
        self.scenario_table.display_load_table()
        self.scenario_table.display_hvdc_table()
        return 


class Table_Show(QWidget):
    def __init__(self, tab_raw, simulator, text_output):
        super(Table_Show, self).__init__()
        self.text_output = text_output
        self.tab_raw = tab_raw
        self.simulator = simulator
        self.simulator.load_dynamic_data('.\\参数设置\\bench_shandong_change_with_gov.dyr', 'PSS/E')
        return                                   

    def display_bus_table(self):  #显示母线数据
        buses = self.simulator.get_all_buses()
        self.tab_raw_bus = QTableWidget()
        self.tab_raw.addTab(self.tab_raw_bus, '母线')
        rows = len(buses)
        header = ['母线号', '母线名', '类型', '区域', '基准电压', '电压/pu', '相角/deg']  
        colunms = len(header)
        self.tab_raw_bus.setRowCount(rows)
        self.tab_raw_bus.setColumnCount(colunms)
        self.tab_raw_bus.setHorizontalHeaderLabels(header)
        for i in range(colunms):
            self.tab_raw_bus.setColumnWidth(i, 150)
            
        for i in range(rows):
            self.tab_raw_bus.setItem(i, 0, QTableWidgetItem(str(buses[i])))
            
            NAME = self.simulator.get_bus_data(buses[i], 'S', 'NAME')
            self.tab_raw_bus.setItem(i, 1, QTableWidgetItem(NAME))
            
            TYPE = self.simulator.get_bus_data(buses[i], 'I', 'TYPE')
            self.tab_raw_bus.setItem(i, 2, QTableWidgetItem(str(TYPE)))
            
            AREA = self.simulator.get_bus_data(buses[i], 'I', 'AREA')
            self.tab_raw_bus.setItem(i, 3, QTableWidgetItem(str(AREA))) 
 
            VBASE_KV = self.simulator.get_bus_data(buses[i], 'F', 'VBASE_KV')
            self.tab_raw_bus.setItem(i, 4, QTableWidgetItem(str(VBASE_KV))) 
            
            V_PU = self.simulator.get_bus_data(buses[i], 'F', 'V_PU')
            V_PU = round(V_PU, 4)
            self.tab_raw_bus.setItem(i, 5, QTableWidgetItem(str(V_PU))) 

            ANGLE_DEG = self.simulator.get_bus_data(buses[i], 'F', 'ANGLE_DEG')
            ANGLE_DEG = round(ANGLE_DEG, 4)
            self.tab_raw_bus.setItem(i, 6, QTableWidgetItem(str(ANGLE_DEG))) 
        #self.tab_raw_bus.verticalScrollBar().setValue(rows)
        return 
        
    def display_generator_table(self):  #加载发电机数据
        self.tab_raw_generator = QTableWidget()
        self.tab_raw.addTab(self.tab_raw_generator, '发电')
        
        generators = self.simulator.get_all_generators()
        rows = len(generators)
        
        header = ['发电机', '母线名', '基准容量/MVA', '有功/MW', '无功/MVAR', '最大出力/MW']
        colunms = len(header)
        self.tab_raw_generator.setColumnCount(colunms)
        self.tab_raw_generator.setRowCount(rows)
        self.tab_raw_generator.setHorizontalHeaderLabels(header)
        
        for i in range(colunms):
            self.tab_raw_generator.setColumnWidth(i, 150)
            
        for i in range(rows):
            self.tab_raw_generator.setItem(i, 0, QTableWidgetItem(str(generators[i])))  
            
            NAME = self.simulator.get_bus_data(generators[i][0], 'S', 'NAME')
            self.tab_raw_generator.setItem(i, 1, QTableWidgetItem(NAME))
            
            MBASE_MVA = self.simulator.get_generator_data(generators[i], 'F', 'MBASE_MVA')
            self.tab_raw_generator.setItem(i, 2, QTableWidgetItem(str(MBASE_MVA))) 

            PGEN_MW = self.simulator.get_generator_data(generators[i], 'F', 'PGEN_MW')
            PGEN_MW = round(PGEN_MW, 4)
            self.tab_raw_generator.setItem(i, 3, QTableWidgetItem(str(PGEN_MW))) 
            
            QGEN_MVAR = self.simulator.get_generator_data(generators[i], 'F', 'QGEN_MVAR')
            QGEN_MVAR = round(QGEN_MVAR, 4)
            self.tab_raw_generator.setItem(i, 4, QTableWidgetItem(str(QGEN_MVAR)))  
            
            PMAX = self.simulator.get_generator_related_model_data(generators[i], 'GOV', 'PMAX')
            PMAX = round(PMAX * MBASE_MVA, 4)
            self.tab_raw_generator.setItem(i, 5, QTableWidgetItem(str(PMAX)))  
        return 
        
    def display_load_table(self):  #加载负荷数据
        self.tab_raw_load = QTableWidget()
        self.tab_raw.addTab(self.tab_raw_load, '负荷')
        
        loads = self.simulator.get_all_loads()
        rows = len(loads)
               
        header = ['负荷', '母线名', '有功/MW', '无功/MVar', '最大切除比例/%']
        colunms = len(header)
        self.tab_raw_load.setColumnCount(colunms)
        self.tab_raw_load.setRowCount(rows)       
        self.tab_raw_load.setHorizontalHeaderLabels(header)
        for i in range(colunms):
            self.tab_raw_load.setColumnWidth(i, 150)
        for i in range(rows):
            self.tab_raw_load.setItem(i, 0, QTableWidgetItem(str(loads[i]))) 
            NAME = self.simulator.get_bus_data(loads[i][0], 'S', 'NAME')
            self.tab_raw_load.setItem(i, 1, QTableWidgetItem(NAME)) 
            
            PP0_MW = self.simulator.get_load_data(loads[i], 'F', 'PP0_MW')
            PP0_MW = round(PP0_MW, 4)
            self.tab_raw_load.setItem(i, 2, QTableWidgetItem(str(PP0_MW)))
            
            QP0_MVAR = self.simulator.get_load_data(loads[i], 'F', 'QP0_MVAR')
            QP0_MVAR = round(QP0_MVAR, 4)
            self.tab_raw_load.setItem(i, 3, QTableWidgetItem(str(QP0_MVAR)))  
        return 
        
    def display_hvdc_table(self):   #加载HVDC数据
        self.tab_raw_hvdc = QTableWidget()
        self.tab_raw.addTab(self.tab_raw_hvdc, '直流')

        hvdcs = self.simulator.get_all_hvdcs()
        rows = len(hvdcs)
        
        header = ['直流', '起始母线', '落点母线', 'PDCN_MW', 'IDCN_KA', 'VDCN_KV', 'RCOMP_OHM', '设置闭锁']
        colunms = len(header)
        self.tab_raw_hvdc.setColumnCount(colunms)
        self.tab_raw_hvdc.setRowCount(rows)
        self.tab_raw_hvdc.setHorizontalHeaderLabels(header)
        for i in range(colunms):
            self.tab_raw_hvdc.setColumnWidth(i, 150)
            
        for i in range(rows):
            self.tab_raw_hvdc.setItem(i, 0, QTableWidgetItem(str(hvdcs[i]))) 
            NAME = self.simulator.get_bus_data(hvdcs[i][0], 'S', 'NAME')
            self.tab_raw_hvdc.setItem(i, 1, QTableWidgetItem(NAME))
            
            NAME = self.simulator.get_bus_data(hvdcs[i][1], 'S', 'NAME')
            self.tab_raw_hvdc.setItem(i, 2, QTableWidgetItem(NAME))  
            
            PDCN_MW = self.simulator.get_hvdc_data(hvdcs[0], 'F', 'HVDC', 'PDCN_MW')
            self.tab_raw_hvdc.setItem(i, 3, QTableWidgetItem(str(PDCN_MW)))

            IDCN_KA = self.simulator.get_hvdc_data(hvdcs[0], 'F', 'HVDC', 'IDCN_KA')
            IDCN_KA = round(IDCN_KA, 4)
            self.tab_raw_hvdc.setItem(i, 4, QTableWidgetItem(str(IDCN_KA)))
            
            VDCN_KV = self.simulator.get_hvdc_data(hvdcs[0], 'F', 'HVDC', 'VDCN_KV')
            self.tab_raw_hvdc.setItem(i, 5, QTableWidgetItem(str(VDCN_KV)))
            
            RCOMP_OHM = self.simulator.get_hvdc_data(hvdcs[0], 'F', 'HVDC', 'RCOMP_OHM')
            self.tab_raw_hvdc.setItem(i, 6, QTableWidgetItem(str(RCOMP_OHM)))
        return 
        
    def set_loads_shedding_location(self, path_name):
        data = pd.read_csv(path_name, header=0, engine='python')
        loads_shedding = data['负荷'].values.tolist() 
        for i in range(len(loads_shedding)):
            loads_shedding[i] = eval(loads_shedding[i])
        max_percent = data['最大切除比例'].values.tolist()

        row = self.tab_raw_load.rowCount()
        for i in range(row):
            load = self.tab_raw_load.item(i, 0).text()
            load = eval(load)
            if load in loads_shedding:
                j = loads_shedding.index(load)
                self.tab_raw_load.setItem(i, 4, QTableWidgetItem(str(max_percent[j]*100)))
        self.text_output.append('导入切负荷设置成功!')
        return 

    def set_block_hvdc_location(self, path_name):
        data = pd.read_csv(path_name, header=0, engine='python')
        data = data['直流'].values.tolist()
        
        for i in range(len(data)):
            data[i] = eval(data[i])

        row = self.tab_raw_hvdc.rowCount()    
        for i in range(row):
            hvdc = self.tab_raw_hvdc.item(i, 0).text()
            hvdc = eval(hvdc)
            if hvdc in data:
                self.tab_raw_hvdc.setItem(i, 7, QTableWidgetItem('是'))
        return
 
    def save_loads_shedding_location(self, path):
        row_num = self.tab_raw_load.rowCount() 
        loads_setting = []
        for i in range(row_num):
            max_percent = self.tab_raw_load.item(i, 4)
            if max_percent is None:
                pass 
            else:
                load = self.tab_raw_load.item(i, 0).text()
                load = eval(load)
                try:
                    max_percent = float(max_percent.text())
                    self.text_output.append('设置负荷{}最大切除比例{}%'.format(load, max_percent))
                    loads_setting.append([load, max_percent/100])
                except:
                    self.text_output.append('警告：负荷{}最大切除比例设置错误'.format(load))
                    pass 
                
        with open(path, 'w', newline='') as f: 
            csv_write = csv.writer(f)
            csv_write.writerow(['负荷', '最大切除比例'])
            csv_write.writerows(loads_setting)       
        self.text_output.append('切负荷设置保存成功!')        
        return 
     
    def save_hvdc_blocking_location(self, path):
        row_num = self.tab_raw_hvdc.rowCount() 
        hvdcs_setting = []
        for i in range(row_num):
            block_bool = self.tab_raw_hvdc.item(i, 7)
            if block_bool is None:
                pass  
            else:
                hvdc = self.tab_raw_hvdc.item(i, 0).text()
                hvdc = eval(hvdc)  
                if block_bool.text() == '是':
                    self.text_output.append('设置直流{}闭锁'.format(hvdc))
                    hvdcs_setting.append([hvdc])
                else:
                    pass 
        with open(path, 'w', newline='') as f: 
            csv_write = csv.writer(f)
            csv_write.writerow(['直流'])
            csv_write.writerows(hvdcs_setting)       
        self.text_output.append('直流闭锁设置保存成功!')                          
        return         


class SecurityConstraint(QDialog):
    def __init__(self, text_output):
        super(SecurityConstraint, self).__init__()
        self.setWindowTitle('安全约束设置')
        self.resize(400, 300)  
        self.mainlayout = QGridLayout()
        self.setLayout(self.mainlayout)

        self.init_security_setting()
        self.text_output = text_output
        return 

    def init_security_setting(self):

        frequency_label = QLabel('最低频率/Hz')
        self.min_frequency_spinbox = QDoubleSpinBox()
        self.min_frequency_spinbox.setValue(49.5)
        self.min_frequency_spinbox.setMinimum(49.0)
        self.min_frequency_spinbox.setMaximum(50.0)
        self.min_frequency_spinbox.setSingleStep(0.01)

        voltage_label = QLabel('电压阈值/pu')
        self.min_volatge_spinbox = QDoubleSpinBox()
        self.min_volatge_spinbox.setValue(0.8)
        self.min_volatge_spinbox.setMinimum(0.7)
        self.min_volatge_spinbox.setMaximum(1.0)
        self.min_volatge_spinbox.setSingleStep(0.01)

        voltage_time_label = QLabel('低于阈值时间/s')
        self.voltage_time_spinbox = QDoubleSpinBox()
        self.voltage_time_spinbox.setValue(0.5)
        self.voltage_time_spinbox.setMinimum(0.0)
        self.voltage_time_spinbox.setMaximum(1.0)
        self.voltage_time_spinbox.setSingleStep(0.01)     

        angle_label  = QLabel('最大功角差/deg')
        self.max_angle_spinbox = QSpinBox()
        self.max_angle_spinbox.setMaximum(180)
        self.max_angle_spinbox.setMinimum(10)
        self.max_angle_spinbox.setValue(160)
        self.max_angle_spinbox.setSingleStep(10)   

        self.save_button = QPushButton('保存')
        self.save_button.clicked.connect(self.save_system_security_setting)     
        
        self.mainlayout.addWidget(frequency_label, 0, 0)
        self.mainlayout.addWidget(self.min_frequency_spinbox, 0, 1)
        self.mainlayout.addWidget(voltage_label, 1, 0)
        self.mainlayout.addWidget(self.min_volatge_spinbox, 1, 1)
        self.mainlayout.addWidget(voltage_time_label, 1, 2)
        self.mainlayout.addWidget(self.voltage_time_spinbox, 1, 3)
        self.mainlayout.addWidget(angle_label, 2, 0)
        self.mainlayout.addWidget(self.max_angle_spinbox, 2, 1)        
        self.mainlayout.addWidget(self.save_button, 4, 3)
        return                 

    def save_system_security_setting(self):
        min_frequency = self.min_frequency_spinbox.value()
        min_voltage = self.min_volatge_spinbox.value()
        voltage_time = self.voltage_time_spinbox.value()
        max_angle = self.max_angle_spinbox.value()

        with open('.\\参数设置\\系统安全约束.csv', 'w', newline='') as f: 
            csv_write = csv.writer(f)
            csv_write.writerow(['最低频率', '电压阈值', '电压时间', '最大功角'])
            csv_write.writerow([min_frequency, min_voltage, voltage_time, max_angle])  
        self.text_output.append('设置最低频率：{}Hz'.format(min_frequency))
        self.text_output.append('设置电压阈值：{}pu'.format(min_voltage))
        self.text_output.append('设置低于电压阈值时间：{}s'.format(voltage_time))
        self.text_output.append('设置最大功角差：{}deg'.format(max_angle))
        return 