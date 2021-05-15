# 基础模型训练界面
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
from ModelDBN import DBNCLAS
from SampleGeneration import SGB 
from DMADE import ADELSM



class UI_SampleGeneration(QDialog):
    def __init__(self):
        super(UI_SampleGeneration, self).__init__()
        self.setWindowTitle('样本生成')
        self.resize(600, 400)
        
        self.mainlayout = QGridLayout()
        self.setLayout(self.mainlayout)
        
        self.parameter_group = QGroupBox('参数设置')
        self.process_group = QGroupBox('生成过程')
        
        self.mainlayout.addWidget(self.parameter_group, 0, 0)
        self.mainlayout.addWidget(self.process_group, 0, 1)
        
        self.init_parameter()
        self.init_process()
        return 
        
    def init_parameter(self):
        self.parameter_layout = QGridLayout()
        self.parameter_group.setLayout(self.parameter_layout)        

        scenario_label = QLabel('目标场景')
        scenario_button = QPushButton('选择')
        scenario_button.clicked.connect(self.choose_scenario)
        self.scenario_line = QLineEdit()
        self.scenario_line.setText('.\\运行方式\\example_scenario_source.raw')
        self.parameter_layout.addWidget(scenario_label, 0, 0)
        self.parameter_layout.addWidget(self.scenario_line, 0, 1, 1, 3)  
        self.parameter_layout.addWidget(scenario_button, 0, 4)  
        
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
        self.parameter_layout.addWidget(pallel_num_label, 1, 3)
        self.parameter_layout.addWidget(self.pallel_num_spinbox, 1, 4) 

        test_data_percent_label = QLabel('测试集比例') 
        self.test_data_percent_spinbox = QDoubleSpinBox()
        self.test_data_percent_spinbox.setValue(0.2)
        self.test_data_percent_spinbox.setMinimum(0.0)
        self.test_data_percent_spinbox.setMaximum(1.0)
        self.test_data_percent_spinbox.setSingleStep(0.05)   

        self.parameter_layout.addWidget(test_data_percent_label, 2, 0)
        self.parameter_layout.addWidget(self.test_data_percent_spinbox, 2, 1) 

        self.check_button = QPushButton('无控制校验')
        self.check_button.clicked.connect(self.check_scenario)
        self.parameter_layout.addWidget(self.check_button, 3, 1)
      
        self.button_generation = QPushButton('生成')
        self.parameter_layout.addWidget(self.button_generation, 3, 4)
        self.button_generation.clicked.connect(self.generate_samples)
        return 

    def init_process(self):
        self.process_layout = QGridLayout()
        self.process_group.setLayout(self.process_layout)
        self.process_textedit = QTextEdit()
        self.process_textedit.setPlainText('样本生成过程记录')
        self.process_layout.addWidget(self.process_textedit, 0, 0)

        self.sample_thread = SampleGenerationThread()#实例化线程对象
        self.sample_thread.finish_signal.connect(self.finish_generation_process)
        self.sample_thread.ckeck_signal.connect(self.finish_checking_scenario)
        self.sample_thread.set_output(self.process_textedit)
        return        
        
        
    def generate_samples(self):
        sample_num = self.sample_num_spinbox.value()
        self.process_textedit.append('设置样本数{}'.format(sample_num))
        pallel_num = self.pallel_num_spinbox.value()
        self.process_textedit.append('设置并行数{}'.format(pallel_num))
        test_data_percent = self.test_data_percent_spinbox.value()
        self.process_textedit.append('设置测试集比例{}'.format(test_data_percent))

        scenario = self.scenario_line.text()
        if len(scenario) == 0:
            self.process_textedit.append('未选择场景')
        self.process_textedit.append('选择未来场景{}'.format(scenario))
        QApplication.processEvents() 
        self.sample_thread.set_generation_parameter({'场景': scenario, '样本数': sample_num, '并行数': pallel_num, '测试集比例': test_data_percent})    
        self.button_generation.setEnabled(False)
        self.sample_thread.set_task_signal(True)
        self.sample_thread.start()
        
        return 
        
    def finish_generation_process(self, str_singal):
        self.button_generation.setEnabled(True)
        self.process_textedit.append('样本生成完毕')
        return 

    def choose_scenario(self):
        openfile_name = QFileDialog.getOpenFileName(self, '场景', '' ,'file(*.raw)')
        if len(openfile_name[0]) == 0:
            return
                
        path_name = openfile_name[0]
        if len(path_name) == 0:
            return
        self.scenario_line.setText(path_name)  
        return        
    
    def check_scenario(self):
        scenario = self.scenario_line.text()
        if len(scenario) == 0:
            self.process_textedit.append('未选择场景')
        self.process_textedit.append('选择未来场景{}'.format(scenario))
        QApplication.processEvents() 
        self.sample_thread.set_generation_parameter({'场景': scenario})    
        self.sample_thread.set_task_signal(False)
        self.check_button.setEnabled(False)
        self.sample_thread.start()

    def finish_checking_scenario(self, str_singal):
        self.check_button.setEnabled(True)
        self.process_textedit.append('场景校验完毕')
        return 

class SampleGenerationThread(QThread):
    finish_signal = pyqtSignal(str)  
    ckeck_signal = pyqtSignal(str)
    def __init__(self):
        super(SampleGenerationThread, self).__init__()
        self.task = True
        return 

    def set_task_signal(self, signal):
        self.task = signal 

    def set_generation_parameter(self, parameter):
        self.parameter = parameter
        return 

    def set_output(self, process_textedit):
        self.process_textedit = process_textedit
        return 

    def run(self): #线程执行函数
        if self.task is False:
            model = SGB()
            model.set_load_shedding_location('.\\参数设置\\切负荷站.csv')
            model.set_blocking_hvdc_location('.\\参数设置\\闭锁直流.csv')        
            model.load_future_scenario(self.parameter['场景'], '.\\参数设置\\bench_shandong_change_with_gov.dyr')  
            min_frequency = model.check_scenario()
            self.process_textedit.append('无控制下直流闭锁故障系统最低频率{}Hz'.format(min_frequency))            
            self.ckeck_signal.emit('测试完成')
        else:
            model = SGB(sample_num=self.parameter['样本数'], pallel_num=self.parameter['并行数'], test_size=self.parameter['测试集比例'])
            model.set_load_shedding_location('.\\参数设置\\切负荷站.csv')
            model.set_blocking_hvdc_location('.\\参数设置\\闭锁直流.csv')        
            model.load_future_scenario(self.parameter['场景'], '.\\参数设置\\bench_shandong_change_with_gov.dyr')  
            model.generate_load_shedding_sample_with_parallel_method(self.process_textedit)
            self.finish_signal.emit('训练完成')
        return


class ModelTraining(QDialog):
    def __init__(self):
        super(ModelTraining, self).__init__()
        self.setWindowTitle('模型训练')
        self.resize(800, 600)
        
        self.mainlayout = QGridLayout()
        self.setLayout(self.mainlayout)
        
        self.training_group = QGroupBox('模型训练')
        self.test_group = QGroupBox('模型测试')
        self.test_result_group = QGroupBox('操作信息')

        self.mainlayout.addWidget(self.training_group, 0, 0, 2, 1)
        self.mainlayout.addWidget(self.test_group, 0, 1) 
        self.mainlayout.addWidget(self.test_result_group, 1, 1) 
        self.init_training_group()
        self.init_test_group()

        self.init_training_data_group()
        self.init_test_data_group()
        self.init_training_parameter_group()
        self.init_test_result_group()
        return

    def init_training_group(self):
        self.training_group_layout = QGridLayout()
        self.training_group.setLayout(self.training_group_layout)

        self.training_data_group = QGroupBox('训练数据')
        self.training_parameter_group = QGroupBox('训练参数设置')
        self.training_button = QPushButton('训练')
        self.training_button.clicked.connect(self.train_dbn_model)

        self.training_group_layout.addWidget(self.training_data_group, 0, 0, 1, 4)
        self.training_group_layout.addWidget(self.training_parameter_group, 2, 0, 1, 4)
        self.training_group_layout.addWidget(self.training_button, 3, 3)
        return 

    def init_test_group(self):
        self.test_group_layout = QGridLayout()
        self.test_group.setLayout(self.test_group_layout)

        self.test_data_group = QGroupBox('测试数据')
        self.test_button = QPushButton('测试')
        self.test_button.clicked.connect(self.test_dbn_model)

        self.test_group_layout.addWidget(self.test_data_group, 0, 0, 1, 4)
        self.test_group_layout.addWidget(self.test_button, 2, 3)

        return 

    def init_training_data_group(self):
        self.training_data_group_layout = QGridLayout()
        self.training_data_group.setLayout(self.training_data_group_layout)

        training_feature_label = QLabel('特征集')
        self.training_feature_line = QLineEdit()
        self.training_feature_line.setText('.\\训练数据\\x_train.csv')
        training_feature_button = QPushButton('选择')
        training_feature_button.clicked.connect(self.choose_training_feature_data)

        training_label_label = QLabel('标签集')
        self.training_label_line = QLineEdit()
        self.training_label_line.setText('.\\训练数据\\y_train.csv')
        training_label_button = QPushButton('选择')
        training_label_button.clicked.connect(self.choose_training_label_data)

        traing_split_label = QLabel('验证集比例')
        self.traing_split_spinbox = QDoubleSpinBox()
        self.traing_split_spinbox.setValue(0.2)
        self.traing_split_spinbox.setMinimum(0.0)
        self.traing_split_spinbox.setMaximum(0.5)
        self.traing_split_spinbox.setSingleStep(0.05)   
        self.traing_split_spinbox.setDecimals(2)  

        training_model_label = QLabel('训练模型')
        self.training_model_line = QLineEdit()
        self.training_model_line.setText('.\\代理模型\\scenario_model.h5')
        training_model_button = QPushButton('选择')
        training_model_button.clicked.connect(self.choose_training_model)

        self.training_data_group_layout.addWidget(training_feature_label, 0, 0)
        self.training_data_group_layout.addWidget(self.training_feature_line, 0, 1, 1, 2)  
        self.training_data_group_layout.addWidget(training_feature_button, 0, 3)
        self.training_data_group_layout.addWidget(training_label_label, 1, 0)
        self.training_data_group_layout.addWidget(self.training_label_line, 1, 1, 1, 2)
        self.training_data_group_layout.addWidget(training_label_button, 1, 3)
        self.training_data_group_layout.addWidget(traing_split_label, 2, 0)
        self.training_data_group_layout.addWidget(self.traing_split_spinbox, 2, 1)
        self.training_data_group_layout.addWidget(training_model_label, 3, 0)
        self.training_data_group_layout.addWidget(self.training_model_line, 3, 1, 1, 2)
        self.training_data_group_layout.addWidget(training_model_button, 3, 3)
        return 

    def init_test_data_group(self):
        self.test_data_group_layout = QGridLayout()
        self.test_data_group.setLayout(self.test_data_group_layout)

        test_feature_label = QLabel('特征集')
        self.test_feature_line = QLineEdit()
        self.test_feature_line.setText('.\\训练数据\\x_test.csv')
        test_feature_button = QPushButton('选择')
        test_feature_button.clicked.connect(self.choose_test_feature_data)

        test_label_label = QLabel('标签集')
        self.test_label_line = QLineEdit()
        self.test_label_line.setText('.\\训练数据\\y_test.csv')
        test_label_button = QPushButton('选择')
        test_label_button.clicked.connect(self.choose_test_label_data)

        test_model_label = QLabel('测试模型')
        self.test_model_line = QLineEdit()
        self.test_model_line.setText('.\\代理模型\\scenario_model.h5')
        test_model_button = QPushButton('选择')
        test_model_button.clicked.connect(self.choose_test_model)

        self.test_data_group_layout.addWidget(test_feature_label, 0, 0)
        self.test_data_group_layout.addWidget(self.test_feature_line, 0, 1, 1, 2)  
        self.test_data_group_layout.addWidget(test_feature_button, 0, 3)
        self.test_data_group_layout.addWidget(test_label_label, 1, 0)
        self.test_data_group_layout.addWidget(self.test_label_line, 1, 1, 1, 2)
        self.test_data_group_layout.addWidget(test_label_button, 1, 3)
        self.test_data_group_layout.addWidget(test_model_label, 2, 0)
        self.test_data_group_layout.addWidget(self.test_model_line, 2, 1, 1, 2)
        self.test_data_group_layout.addWidget(test_model_button, 2, 3)
        return 

    def init_training_parameter_group(self):
        self.training_parameter_group_layout = QGridLayout()
        self.training_parameter_group.setLayout(self.training_parameter_group_layout) 

        self.layer_structure_group = QGroupBox('模型结构')
        self.layer_structure_group_layout = QGridLayout()
        self.layer_structure_group.setLayout(self.layer_structure_group_layout)

        layer_1_label = QLabel('第一层')
        self.layer_1_spinbox = QSpinBox()
        self.layer_1_spinbox.setMinimum(10)
        self.layer_1_spinbox.setMaximum(200)
        self.layer_1_spinbox.setValue(50)
        self.layer_1_spinbox.setSingleStep(5)        

        layer_2_label = QLabel('第二层')
        self.layer_2_spinbox = QSpinBox()
        self.layer_2_spinbox.setMinimum(10)
        self.layer_2_spinbox.setMaximum(200)
        self.layer_2_spinbox.setValue(10)
        self.layer_2_spinbox.setSingleStep(5) 

        layer_3_label = QLabel('第三层')
        self.layer_3_spinbox = QSpinBox()
        self.layer_3_spinbox.setMinimum(0)
        self.layer_3_spinbox.setMaximum(200)
        self.layer_3_spinbox.setValue(0)
        self.layer_3_spinbox.setSingleStep(5) 

        layer_4_label = QLabel('第四层')
        self.layer_4_spinbox = QSpinBox()
        self.layer_4_spinbox.setMinimum(0)
        self.layer_4_spinbox.setMaximum(200)
        self.layer_4_spinbox.setValue(0)
        self.layer_4_spinbox.setSingleStep(5)                         

        self.layer_structure_group_layout.addWidget(layer_1_label, 0, 0)
        self.layer_structure_group_layout.addWidget(self.layer_1_spinbox, 0, 1)
        self.layer_structure_group_layout.addWidget(QLabel(), 0, 2)
        self.layer_structure_group_layout.addWidget(layer_2_label, 0, 3)
        self.layer_structure_group_layout.addWidget(self.layer_2_spinbox, 0, 4)
        self.layer_structure_group_layout.addWidget(layer_3_label, 1, 0)
        self.layer_structure_group_layout.addWidget(self.layer_3_spinbox, 1, 1)
        self.layer_structure_group_layout.addWidget(QLabel(), 1, 2)
        self.layer_structure_group_layout.addWidget(layer_4_label, 1, 3)
        self.layer_structure_group_layout.addWidget(self.layer_4_spinbox, 1, 4)

        self.rbm_training_group = QGroupBox('RBM预训练')
        self.rbm_training_group_layout = QGridLayout()
        self.rbm_training_group.setLayout(self.rbm_training_group_layout)

        rbm_epochs_label = QLabel('学习代数')
        self.rbm_epochs_spinbox = QSpinBox()
        self.rbm_epochs_spinbox.setMinimum(10)
        self.rbm_epochs_spinbox.setMaximum(100)
        self.rbm_epochs_spinbox.setValue(20)
        self.rbm_epochs_spinbox.setSingleStep(5)
        
        rbm_learning_rate_label = QLabel('学习率')
        self.rbm_learning_rate_spinbox = QDoubleSpinBox()
        self.rbm_learning_rate_spinbox.setDecimals(4)
        self.rbm_learning_rate_spinbox.setMinimum(0.0)
        self.rbm_learning_rate_spinbox.setMaximum(1.0)
        self.rbm_learning_rate_spinbox.setValue(0.0002)
        self.rbm_learning_rate_spinbox.setSingleStep(0.0001)             

        rbm_batch_size_label = QLabel('批数')
        self.rbm_batch_size_spinbox = QSpinBox()
        self.rbm_batch_size_spinbox.setMinimum(10)
        self.rbm_batch_size_spinbox.setMaximum(500)
        self.rbm_batch_size_spinbox.setValue(50)
        self.rbm_batch_size_spinbox.setSingleStep(10)

        self.rbm_training_group_layout.addWidget(rbm_epochs_label, 0, 0)
        self.rbm_training_group_layout.addWidget(self.rbm_epochs_spinbox, 0, 1)
        self.rbm_training_group_layout.addWidget(QLabel(), 0, 2)
        self.rbm_training_group_layout.addWidget(rbm_learning_rate_label, 0, 3)
        self.rbm_training_group_layout.addWidget(self.rbm_learning_rate_spinbox, 0, 4)
        self.rbm_training_group_layout.addWidget(rbm_batch_size_label, 1, 0)
        self.rbm_training_group_layout.addWidget(self.rbm_batch_size_spinbox, 1, 1)

        self.finetune_group = QGroupBox('微调设置')
        self.finetune_group_layout = QGridLayout()
        self.finetune_group.setLayout(self.finetune_group_layout)

        nn_epochs_label = QLabel('学习代数')
        self.nn_epochs_spinbox = QSpinBox()
        self.nn_epochs_spinbox.setMinimum(10)
        self.nn_epochs_spinbox.setMaximum(2000)
        self.nn_epochs_spinbox.setValue(1000)
        self.nn_epochs_spinbox.setSingleStep(100)

        nn_learning_rate_label = QLabel('学习率')
        self.nn_learning_rate_spinbox = QDoubleSpinBox()
        self.nn_learning_rate_spinbox.setDecimals(5) 
        self.nn_learning_rate_spinbox.setMinimum(0.0)
        self.nn_learning_rate_spinbox.setMaximum(1.0)
        self.nn_learning_rate_spinbox.setValue(0.01)
        self.nn_learning_rate_spinbox.setSingleStep(0.00001)                      

        regularizer_l2_label = QLabel('正则化')
        self.regularizer_l2_spinbox = QDoubleSpinBox()
        self.regularizer_l2_spinbox.setDecimals(6) 
        self.regularizer_l2_spinbox.setMinimum(0.0)
        self.regularizer_l2_spinbox.setMaximum(1.0)
        self.regularizer_l2_spinbox.setValue(0.0001)
        self.regularizer_l2_spinbox.setSingleStep(0.00001)                

        nn_decay_rate_label = QLabel('学习率下降')
        self.nn_decay_rate_spinbox = QDoubleSpinBox()
        self.nn_decay_rate_spinbox.setDecimals(5)
        self.nn_decay_rate_spinbox.setMinimum(0.0)
        self.nn_decay_rate_spinbox.setMaximum(0.1)
        self.nn_decay_rate_spinbox.setValue(0.00010)
        self.nn_decay_rate_spinbox.setSingleStep(0.00001)        
        
        nn_batch_size = QLabel('批数')
        self.nn_batch_size_spinbox = QSpinBox()
        self.nn_batch_size_spinbox.setMinimum(10)
        self.nn_batch_size_spinbox.setMaximum(500)
        self.nn_batch_size_spinbox.setValue(50)
        self.nn_batch_size_spinbox.setSingleStep(10)

        self.finetune_group_layout.addWidget(nn_epochs_label, 0, 0)
        self.finetune_group_layout.addWidget(self.nn_epochs_spinbox, 0, 1)
        self.finetune_group_layout.addWidget(QLabel(), 0, 2)
        self.finetune_group_layout.addWidget(nn_learning_rate_label, 0, 3)
        self.finetune_group_layout.addWidget(self.nn_learning_rate_spinbox, 0, 4)
        self.finetune_group_layout.addWidget(regularizer_l2_label, 1, 0)
        self.finetune_group_layout.addWidget(self.regularizer_l2_spinbox, 1, 1)
        self.finetune_group_layout.addWidget(QLabel(), 1, 2)
        self.finetune_group_layout.addWidget(nn_decay_rate_label, 1, 3)
        self.finetune_group_layout.addWidget(self.nn_decay_rate_spinbox, 1, 4)
        self.finetune_group_layout.addWidget(nn_batch_size, 2, 0)
        self.finetune_group_layout.addWidget(self.nn_batch_size_spinbox, 2, 1)

        self.training_parameter_group_layout.addWidget(self.layer_structure_group, 0, 0, 2, 5)
        self.training_parameter_group_layout.addWidget(self.rbm_training_group, 2, 0, 1, 5)
        self.training_parameter_group_layout.addWidget(self.finetune_group, 3, 0, 2, 5)
        return 

    def init_test_result_group(self):
        self.test_result_group_layout = QGridLayout()
        self.test_result_group.setLayout(self.test_result_group_layout)
        self.result_textedit = QTextEdit()
        self.result_textedit.setPlainText('模型训练测试结果')
        self.test_result_group_layout.addWidget(self.result_textedit, 0, 0)  

        self.model_training_thread = ModelTrainingThread()
        self.model_training_thread.finish_signal.connect(self.finish_model_training) 
        self.model_training_thread.process_signal.connect(self.display_training_process)  
    
    
    def display_training_process(self, process_str):
        self.result_textedit.append(process_str)
        self.result_textedit.verticalScrollBar().setValue(self.result_textedit.verticalScrollBar().maximum())

    def choose_training_feature_data(self):
        openfile_name = QFileDialog.getOpenFileName(self, '训练特征集', '' ,'file(*.csv)')
        if len(openfile_name[0]) == 0:
            return
                
        path_name = openfile_name[0]
        if len(path_name) == 0:
            return
        self.training_feature_line.setText(path_name)   

    def choose_training_label_data(self):
        openfile_name = QFileDialog.getOpenFileName(self, '训练标签', '' ,'file(*.csv)')
        if len(openfile_name[0]) == 0:
            return
                
        path_name = openfile_name[0]
        if len(path_name) == 0:
            return
        self.training_label_line.setText(path_name)   
        return 
    
    def choose_training_model(self):
        path_name = QFileDialog.getSaveFileName(self, '保存模型', '' , '(*.h5)')
        if len(path_name[0]) == 0:
            return 
        self.training_model_line.setText(path_name[0])  
        self.test_model_line.setText(path_name[0])       
        return 

    def train_dbn_model(self):
        layer_1 = self.layer_1_spinbox.value()
        layer_2 = self.layer_2_spinbox.value()
        layer_3 = self.layer_3_spinbox.value()
        layer_4 = self.layer_4_spinbox.value()
        
        layer_structure = []
        for layer in [layer_1, layer_2, layer_3, layer_4]:
            if layer == 0:
                break 
            else:
                layer_structure.append(layer)

        rbm_epochs_number = self.rbm_epochs_spinbox.value()
        rbm_learning_rate = self.rbm_learning_rate_spinbox.value()
        rbm_batch_size = self.rbm_batch_size_spinbox.value()

        nn_learning_rate = self.nn_learning_rate_spinbox.value()
        nn_decay_rate = self.nn_decay_rate_spinbox.value()
        regularizer_l2 = self.regularizer_l2_spinbox.value()
        nn_epochs_number = self.nn_epochs_spinbox.value()
        nn_batch_size = self.nn_batch_size_spinbox.value()

        val_split = self.traing_split_spinbox.value()

        x_train_path = self.training_feature_line.text()
        y_train_path = self.training_label_line.text()
        if len(x_train_path) == 0 or len(y_train_path) == 0:
            return         

        self.model = DBNCLAS(self.result_textedit)  # 初始化
        self.model.load_train_data(x_train_path, y_train_path, val_split)  # 加载训练数据和测试数据
        self.model.set_hidden_layer_structure(layer_structure)
        
        self.model.set_rbm_epochs_number(rbm_epochs_number)  # RBM学习次数
        self.model.set_rbm_learning_rate(rbm_learning_rate)  # RBM学习率
        self.model.set_rbm_batch_size(rbm_batch_size)  # RBM批处理

        self.model.set_nn_learning_rate(nn_learning_rate)  # 神经网络学习率
        self.model.set_nn_decay_rate(nn_decay_rate)  # 学习率下降参数
        self.model.set_hidden_layer_regularizer_l2(regularizer_l2)   # 设置l2正则化系数
        self.model.set_nn_epochs_number(nn_epochs_number)  # 学习次数
        self.model.set_nn_batch_size(nn_batch_size)  # 学习批处理个数

        self.training_button.setEnabled(False)
        self.model_training_thread.set_training_model(self.model)
        self.result_textedit.append('模型正在训练中…………')

        self.model_training_thread.start()
        return 

    def finish_model_training(self, finish_sig):
        self.training_button.setEnabled(True)
        self.result_textedit.append('模型训练完成…………')
        model_path = self.training_model_line.text()
        if len(model_path) != 0:
            self.model.save_model(model_path)  # 保存模型
        return 

    def choose_test_feature_data(self):
        openfile_name = QFileDialog.getOpenFileName(self, '测试特征集', '' ,'file(*.csv)')
        if len(openfile_name[0]) == 0:
            return
                
        path_name = openfile_name[0]
        if len(path_name) == 0:
            return
        self.test_feature_line.setText(path_name)   
        return 

    def choose_test_label_data(self):
        openfile_name = QFileDialog.getOpenFileName(self, '测试标签', '' ,'file(*.csv)')
        if len(openfile_name[0]) == 0:
            return
                
        path_name = openfile_name[0]
        if len(path_name) == 0:
            return
        self.test_label_line.setText(path_name)   
        return 
    
    def choose_test_model(self):
        openfile_name = QFileDialog.getOpenFileName(self, '测试模型', '' ,'file(*.h5)')
        if len(openfile_name[0]) == 0:
            return      
        path_name = openfile_name[0]
        self.test_model_line.setText(path_name)       
        return 
    
    def test_dbn_model(self):
        model_path = self.test_model_line.text()
        if len(model_path) == 0:
            return 

        x_test_path = self.test_feature_line.text()
        y_test_path = self.test_label_line.text()
        if len(x_test_path) == 0 or len(y_test_path) == 0:
            return             
        self.model = DBNCLAS()
        self.model.load_test_data(x_test_path, y_test_path)
        self.model.load_model(model_path)
        acc, acc_mat = self.model.get_model_accuracy()
        self.result_textedit.append('模型总体精度: {}'.format(acc))
        self.result_textedit.append('模型混淆矩阵: \n{}'.format(acc_mat))
        return     

class ModelTrainingThread(QThread):#线程类
    finish_signal = pyqtSignal(str)  
    process_signal = pyqtSignal(str)
    def __init__(self):
        super(ModelTrainingThread, self).__init__()
        return 
        
    def set_training_model(self, model):
        self.model = model
        return 

    def run(self): #线程执行函数
        self.model.pretrain()  # 预训练
        history = self.model.fine_tune()  # 微调
        for i in range(len(history.history['acc'])):
            self.process_signal.emit('训练{}训练集损失{:.4f}--准确率{:.4f}'.format(i+1, 
                history.history['loss'][i], history.history['acc'][i]))

        plt.figure(1)        
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('模型训练损失')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'])

        plt.figure(2)        
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model training acc')
        plt.ylabel('acc')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'])
        plt.show()
        self.finish_signal.emit('训练完成')
       
       
class AMSOD(QDialog):
    def __init__(self):
        super(AMSOD, self).__init__()
        self.setWindowTitle('代理辅助模型优化计算')
        self.resize(800, 600)

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
        return 
        
    def init_parameter(self):
        self.parameter_layout = QGridLayout()
        self.parameter_group.setLayout(self.parameter_layout)


        scenario_button = QPushButton('选择场景')
        scenario_button.clicked.connect(self.choose_scenario)
        self.scenario_line = QLineEdit()
        self.scenario_line.setText('.\\运行方式\\example_scenario_source.raw')
        self.parameter_layout.addWidget(scenario_button, 0, 0)
        self.parameter_layout.addWidget(self.scenario_line, 0, 1, 1, 4)    

        ass_model_button = QPushButton('选择模型')
        ass_model_button.clicked.connect(self.choose_ass_model)
        self.ass_model_line = QLineEdit()
        self.ass_model_line.setText('.\\代理模型\\model_source_example.h5')
        self.parameter_layout.addWidget(ass_model_button, 1, 0)
        self.parameter_layout.addWidget(self.ass_model_line, 1, 1, 1, 4)        

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
        return 
    
    def init_optimize_result(self):
        self.optimize_result_layout = QGridLayout()
        self.result_group.setLayout(self.optimize_result_layout)
        
        self.optimize_result_table = QTableWidget()
        self.optimize_result_table.setColumnCount(5)
        self.optimize_result_table.setRowCount(10)   
        header = ['切负荷点', '母线名', '负荷量/MW', '切除比例/%', '切除量/MW']
        self.optimize_result_table.setHorizontalHeaderLabels(header)
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
        return 
    
    
    def start_optimizer(self):
        ass_model = self.ass_model_line.text()
        if len(ass_model) == 0:
            return 
        raw_file = self.scenario_line.text()
        if len(raw_file) == 0:
            return  

        size = self.size_spinbox.value()
        iter_num = self.iter_num_spinbox.value()
        F = self.F_spinbox.value()
        CR = self.CR_spinbox.value()

        optimizer = ADELSM(self.process_textedit)
        optimizer.load_frequency_classification_prediction_model(ass_model)
        
        optimizer.load_scenario_data(raw_file)
        
        optimizer.set_evolution_parameter('种群数目', size)
        optimizer.set_evolution_parameter('进化代数', iter_num)  
        optimizer.set_evolution_parameter('交叉因子', CR)
        optimizer.set_evolution_parameter('变异因子', F)    
        optimizer.initialize_population()
        
        optimizer.operate_evolution()
        optimizer.plot_evolution_process()
        
        best_individual, loads_p, loads_shedding, loads_name = optimizer.save_best_individual('.\\优化结果\\最优切负荷方案.csv')
        self.display_best_load_shedding_plan(best_individual, loads_p, loads_shedding, loads_name)
        
        min_frequency = optimizer.check_evolution_result(best_individual)
        self.min_frequency_result_line.setText(str(min_frequency) + '  Hz')
        return 

    def choose_ass_model(self):
        openfile_name = QFileDialog.getOpenFileName(self, '代理模型', '' ,'file(*.h5)')
        if len(openfile_name[0]) == 0:
            return
                
        path_name = openfile_name[0]
        if len(path_name) == 0:
            return
        self.ass_model_line.setText(path_name)         
        return 

    def choose_scenario(self):
        openfile_name = QFileDialog.getOpenFileName(self, '场景', '' ,'file(*.raw)')
        if len(openfile_name[0]) == 0:
            return
                
        path_name = openfile_name[0]
        if len(path_name) == 0:
            return
        self.scenario_line.setText(path_name)  
        return 

    def display_best_load_shedding_plan(self, best_individual, loads_p, loads_shedding, loads_name):
        shedding_power = best_individual * loads_p
        power = round(np.sum(shedding_power), 4)
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

        


