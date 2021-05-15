from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import sys 
sys.path.append('.\\code')
from SampleGeneration import SGTL
from ModelDBN import DBNTL
from DMADE import ADELSM
import pandas as pd 
import numpy as np 


class UI_SGTL(QDialog):
    def __init__(self):
        super(UI_SGTL, self).__init__()   
        self.setWindowTitle('未来方式下样本生成') 
        self.resize(800, 600)

        self.mainlayout = QGridLayout()
        self.setLayout(self.mainlayout)
        self.parameter_group = QGroupBox('参数设置')
        self.process_group = QGroupBox('生成过程')
        self.mainlayout.addWidget(self.parameter_group, 0, 0)
        self.mainlayout.addWidget(self.process_group, 0, 1)   

        self.init_parameter()
        self.init_process()

        self.FSGT = SampleGenerationThread()
        self.FSGT.finish_signal.connect(self.finish_sample_generation)
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
        self.sample_num_spinbox.setMinimum(100)
        self.sample_num_spinbox.setMaximum(1000)
        self.sample_num_spinbox.setValue(10)
        self.sample_num_spinbox.setSingleStep(50)    

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
        self.parameter_layout.addWidget(self.generation_button, 4, 3)        
        return 

    def init_process(self):
        self.process_layout = QGridLayout()
        self.process_group.setLayout(self.process_layout)
        self.process_textedit = QTextEdit()
        self.process_textedit.setPlainText('样本生成过程')
        self.process_layout.addWidget(self.process_textedit, 0, 0)
        return 

    def start_sample_generation(self):
        best_scheme = self.current_best_scheme_line.text()
        sample_scale = self.sample_scale_spinbox.value()
        sample_num = self.sample_num_spinbox.value()
        pallel_num = self.pallel_num_spinbox.value()
        deviation_num = self.deviation_spinbox.value()

        future_scenario_raw = self.future_scenario_line.text()
        if len(future_scenario_raw) == 0:
            self.process_textedit.append('未选择未来场景')
            return 

        FSG = SGTL()
        FSG.set_best_scheme(best_scheme)
        
        FSG.set_parameter_data('分布系数', sample_scale)
        FSG.set_parameter_data('偏移系数', deviation_num)
        FSG.set_parameter_data('样本数', sample_num)
        FSG.set_parameter_data('并行数', pallel_num)

        FSG.set_future_scenario(future_scenario_raw)
        
        self.process_textedit.append('加载未来场景完毕{}'.format(future_scenario_raw))
        self.process_textedit.append('设置并行数{}'.format(pallel_num))
        self.process_textedit.append('设置样本数{}'.format(sample_num))
        self.process_textedit.append('设置当前最优方案{}'.format(best_scheme))
        self.process_textedit.append('设置分布系数{}'.format(sample_scale))
        self.process_textedit.append('设置偏移系数{}'.format(deviation_num))
        QApplication.processEvents() 
        self.generation_button.setEnabled(False)
        self.FSGT.set_model(FSG, self.process_textedit)
        self.FSGT.start()
        return  

    def finish_sample_generation(self, singal):
        self.process_textedit.append('样本生成完成')
        self.generation_button.setEnabled(True)
        return 

    def check_origin_scheme(self):
        self.check_origin_scheme_button.setEnabled(False)
        QApplication.processEvents() 
        best_scheme = self.current_best_scheme_line.text()
        future_scenario_raw = self.future_scenario_line.text()
        if len(future_scenario_raw) == 0:
            self.process_textedit.append('未选择未来场景')
            return 

        FSG = SGTL()
        FSG.set_future_scenario(future_scenario_raw)
        FSG.set_best_scheme(best_scheme) 
        min_frequency = FSG.check_origin_scheme()
        self.process_textedit.append('原方案下系统的最低频率:{}'.format(min_frequency)) 
        self.check_origin_scheme_button.setEnabled(True)
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

class SampleGenerationThread(QThread):#线程类
    finish_signal = pyqtSignal(str)  #自定义信号对象。参数str就代表这个信号可以传一个字符串
    def __init__(self):
        super(SampleGenerationThread, self).__init__()
        return 
        
    def set_model(self, model, process_textedit):
        self.model = model
        self.process_textedit = process_textedit
        return 

    def run(self): #线程执行函数
        self.model.generate_new_scenario_load_shedding_sample()
        self.model.generate_load_shedding_sample_with_parallel_method_in_future_scenario(self.process_textedit)
        self.finish_signal.emit('训练完成')
        return

class MTTL(QDialog):
    def __init__(self):
        super(MTTL, self).__init__()   
        self.resize(600, 600)
        self.setWindowTitle('模型迁移')
        self.mainlayout = QGridLayout()
        self.setLayout(self.mainlayout)
        
        self.training_group = QGroupBox('模型训练')
        self.train_prosess_group = QGroupBox('训练过程信息')

        self.mainlayout.addWidget(self.training_group, 0, 0) 
        self.mainlayout.addWidget(self.train_prosess_group, 1, 0)
         
        self.init_training_group()
        self.init_training_data_group()
        self.init_training_parameter_group()
        
        self.init_train_prosess_group()        
        
        return 
        

    def init_training_group(self):
        self.training_group_layout = QGridLayout()
        self.training_group.setLayout(self.training_group_layout)

        self.training_data_group = QGroupBox('训练数据')
        self.training_parameter_group = QGroupBox('微调参数设置')

        self.training_group_layout.addWidget(self.training_data_group, 0, 0)
        self.training_group_layout.addWidget(self.training_parameter_group, 1, 0)
        return 

    def init_training_data_group(self):
        self.training_data_group_layout = QGridLayout()
        self.training_data_group.setLayout(self.training_data_group_layout)

        training_feature_label = QLabel('特征集')
        self.training_feature_line = QLineEdit()
        self.training_feature_line.setText('.\\训练数据\\x_train_tl.csv')
        training_feature_button = QPushButton('选择')
        training_feature_button.clicked.connect(self.choose_training_feature_data)

        training_label_label = QLabel('标签集')
        self.training_label_line = QLineEdit()
        self.training_label_line.setText('.\\训练数据\\y_train_tl.csv')
        training_label_button = QPushButton('选择')
        training_label_button.clicked.connect(self.choose_training_label_data)

        traing_split_label = QLabel('验证集比例')
        self.traing_split_spinbox = QDoubleSpinBox()
        self.traing_split_spinbox.setValue(0.2)
        self.traing_split_spinbox.setMinimum(0.0)
        self.traing_split_spinbox.setMaximum(0.5)
        self.traing_split_spinbox.setSingleStep(0.05)   
        self.traing_split_spinbox.setDecimals(2)  

        source_model_label = QLabel('源模型')
        self.source_model_line = QLineEdit()
        self.source_model_line.setText('.\\代理模型\\level_100_cla.h5')
        source_model_button = QPushButton('选择')
        source_model_button.clicked.connect(self.choose_source_model)

        target_model_label = QLabel('目标模型')
        self.target_model_line = QLineEdit()
        self.target_model_line.setText('.\\代理模型\\target_model.h5')
        target_model_button = QPushButton('选择')
        target_model_button.clicked.connect(self.choose_target_model)

        self.training_data_group_layout.addWidget(training_feature_label, 0, 0)
        self.training_data_group_layout.addWidget(self.training_feature_line, 0, 1, 1, 3)  
        self.training_data_group_layout.addWidget(training_feature_button, 0, 4)
        self.training_data_group_layout.addWidget(training_label_label, 1, 0)
        self.training_data_group_layout.addWidget(self.training_label_line, 1, 1, 1, 3)
        self.training_data_group_layout.addWidget(training_label_button, 1, 4)
        self.training_data_group_layout.addWidget(traing_split_label, 2, 0)
        self.training_data_group_layout.addWidget(self.traing_split_spinbox, 2, 1)
        self.training_data_group_layout.addWidget(source_model_label, 3, 0)
        self.training_data_group_layout.addWidget(self.source_model_line, 3, 1, 1, 3)
        self.training_data_group_layout.addWidget(source_model_button, 3, 4)

        self.training_data_group_layout.addWidget(target_model_label, 4, 0)
        self.training_data_group_layout.addWidget(self.target_model_line, 4, 1, 1, 3)
        self.training_data_group_layout.addWidget(target_model_button, 4, 4)
        return 

    def init_training_parameter_group(self):
        self.finetune_group_layout = QGridLayout()
        self.training_parameter_group.setLayout(self.finetune_group_layout)

        nn_epochs_label = QLabel('学习代数')
        self.nn_epochs_spinbox = QSpinBox()
        self.nn_epochs_spinbox.setMinimum(10)
        self.nn_epochs_spinbox.setMaximum(2000)
        self.nn_epochs_spinbox.setValue(100)
        self.nn_epochs_spinbox.setSingleStep(100)

        nn_learning_rate_label = QLabel('学习率')
        self.nn_learning_rate_spinbox = QDoubleSpinBox()
        self.nn_learning_rate_spinbox.setDecimals(5) 
        self.nn_learning_rate_spinbox.setMinimum(0.0)
        self.nn_learning_rate_spinbox.setMaximum(1.0)
        self.nn_learning_rate_spinbox.setValue(0.0001)
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
        self.nn_decay_rate_spinbox.setValue(0.0010)
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

        self.training_button = QPushButton('微调')
        self.training_button.clicked.connect(self.fine_tune_dbn_model)
        self.finetune_group_layout.addWidget(self.training_button, 3, 4)
        return

    def init_train_prosess_group(self):
        self.test_result_group_layout = QGridLayout()
        self.train_prosess_group.setLayout(self.test_result_group_layout)
        self.result_textedit = QTextEdit()
        self.result_textedit.setPlainText('训练过程信息')
        self.test_result_group_layout.addWidget(self.result_textedit, 0, 0)        
        return 

    def choose_training_feature_data(self):
        openfile_name = QFileDialog.getOpenFileName(self, '训练特征集', '' ,'file(*.csv)')
        if len(openfile_name[0]) == 0:
            return
                
        path_name = openfile_name[0]
        if len(path_name) == 0:
            return
        self.training_feature_line.setText(path_name)   
        return 

    def choose_training_label_data(self):
        openfile_name = QFileDialog.getOpenFileName(self, '训练标签', '' ,'file(*.csv)')
        if len(openfile_name[0]) == 0:
            return
                
        path_name = openfile_name[0]
        if len(path_name) == 0:
            return
        self.training_label_line.setText(path_name)   
        return 
    
    def choose_source_model(self):
        openfile_name = QFileDialog.getOpenFileName(self, '源模型', '' ,'file(*.h5)')
        if len(openfile_name[0]) == 0:
            return
                
        path_name = openfile_name[0]
        if len(path_name) == 0:
            return
        self.source_model_line.setText(path_name)       
        return

    def choose_target_model(self):
        openfile_name = QFileDialog.getSaveFileName(self, '保存模型', '' ,'file(*.h5)')
        if len(openfile_name[0]) == 0:
            return
                
        path_name = openfile_name[0]
        if len(path_name) == 0:
            return
        self.target_model_line.setText(path_name)        
        return 

    def fine_tune_dbn_model(self):
        source_model_path = self.source_model_line.text()
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

        model = DBNTL()  # 初始化
        model.load_train_data(x_train_path, y_train_path, val_split)  # 加载训练数据和测试数据
        
        model.load_model(source_model_path)
        model.set_nn_learning_rate(nn_learning_rate)  # 神经网络学习率
        model.set_nn_decay_rate(nn_decay_rate)  # 学习率下降参数
        model.set_hidden_layer_regularizer_l2(regularizer_l2)   # 设置l2正则化系数
        model.set_nn_epochs_number(nn_epochs_number)  # 学习次数
        model.set_nn_batch_size(nn_batch_size)  # 学习批处理个数

        model.build_target_model()
        history = model.fine_tune()  # 微调
        
        for i in range(len(history.history['acc'])):
            self.result_textedit.append('训练{}训练集损失{:.4f}--准确率{:.4f}--验证集损失{:.4f}--准确率{:.4f}'.format(i+1, 
                history.history['loss'][i], history.history['acc'][i], history.history['val_loss'][i], history.history['val_acc'][i]))
        QApplication.processEvents()
        model_path = self.target_model_line.text()
        if len(model_path) != 0:
            model.save_model(model_path)  # 保存模型

        return 

    def finish_model_training(self, finish_sig):
        self.training_button.setEnabled(True)
        self.result_textedit.append('模型训练完成…………')
        model_path = self.training_model_line.text()
        if len(model_path) != 0:
            self.model.save_model(model_path)  # 保存模型
        return 


class AMSODTL(QDialog):
    def __init__(self):
        super(AMSODTL, self).__init__()
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
        self.scenario_line.setText('.\\运行方式\\example_scenario_target_up.raw')
        self.parameter_layout.addWidget(scenario_button, 0, 0)
        self.parameter_layout.addWidget(self.scenario_line, 0, 1, 1, 4)    

        ass_model_button = QPushButton('选择模型')
        ass_model_button.clicked.connect(self.choose_ass_model)
        self.ass_model_line = QLineEdit()
        self.ass_model_line.setText('.\\代理模型\\model_target_example.h5')
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
        min_frequency = round(min_frequency, 6)
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