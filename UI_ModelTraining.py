# 基础模型训练界面
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
 
import sys 
sys.path.append('.\\code')
from ModelDBN import DBNCLAS, DBNTL


class UI_DBNBM(QDialog):
    def __init__(self):
        super(UI_DBNBM, self).__init__()
        self.setWindowTitle('模型训练')
        self.resize(800, 600)
        self.setWindowIcon(QIcon('.\\logo\\安全.png'))

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

    def init_test_group(self):
        self.test_group_layout = QGridLayout()
        self.test_group.setLayout(self.test_group_layout)

        self.test_data_group = QGroupBox('测试数据')
        self.test_button = QPushButton('测试')
        self.test_button.clicked.connect(self.test_dbn_model)

        self.test_group_layout.addWidget(self.test_data_group, 0, 0, 1, 4)
        self.test_group_layout.addWidget(self.test_button, 2, 3)

    def init_training_data_group(self):
        self.training_data_group_layout = QGridLayout()
        self.training_data_group.setLayout(self.training_data_group_layout)

        training_feature_label = QLabel('特征集')
        self.training_feature_line = QLineEdit()
        self.training_feature_line.setText('.\\训练数据\\x_train_example.csv')
        training_feature_button = QPushButton('选择')
        training_feature_button.clicked.connect(self.choose_training_feature_data)

        training_label_label = QLabel('标签集')
        self.training_label_line = QLineEdit()
        self.training_label_line.setText('.\\训练数据\\y_train_example.csv')
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
         
    def init_test_data_group(self):
        self.test_data_group_layout = QGridLayout()
        self.test_data_group.setLayout(self.test_data_group_layout)

        test_feature_label = QLabel('特征集')
        self.test_feature_line = QLineEdit()
        self.test_feature_line.setText('.\\训练数据\\x_test_example.csv')
        test_feature_button = QPushButton('选择')
        test_feature_button.clicked.connect(self.choose_test_feature_data)

        test_label_label = QLabel('标签集')
        self.test_label_line = QLineEdit()
        self.test_label_line.setText('.\\训练数据\\y_test_example.csv')
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

    def init_test_result_group(self):
        self.test_result_group_layout = QGridLayout()
        self.test_result_group.setLayout(self.test_result_group_layout)
        self.result_textedit = QTextEdit()
        self.result_textedit.setPlainText('模型测试结果')
        acc_label = QLabel('测试集模型精度')
        self.acc_line = QLineEdit()
        self.test_result_group_layout.addWidget(self.result_textedit, 0, 0, 1, 2)  
        self.test_result_group_layout.addWidget(acc_label, 1, 0)
        self.test_result_group_layout.addWidget(self.acc_line, 1, 1)

        self.model_training_thread = ModelTrainingThread()
        self.model_training_thread.finish_signal.connect(self.finish_model_training) 

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
            QMessageBox.warning(self,'警告','未选择训练集', QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)
            return         

        self.model = DBNCLAS()  # 初始化
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

    def finish_model_training(self, finish_sig):
        self.training_button.setEnabled(True)
        self.result_textedit.append('模型训练完成…………')
        model_path = self.training_model_line.text()
        if len(model_path) != 0:
            self.model.save_model(model_path)  # 保存模型

    def choose_test_feature_data(self):
        openfile_name = QFileDialog.getOpenFileName(self, '测试特征集', '' ,'file(*.csv)')
        if len(openfile_name[0]) == 0:
            return
                
        path_name = openfile_name[0]
        if len(path_name) == 0:
            return
        self.test_feature_line.setText(path_name)    

    def choose_test_label_data(self):
        openfile_name = QFileDialog.getOpenFileName(self, '测试标签', '' ,'file(*.csv)')
        if len(openfile_name[0]) == 0:
            return
                
        path_name = openfile_name[0]
        if len(path_name) == 0:
            return
        self.test_label_line.setText(path_name)   
    
    def choose_test_model(self):
        openfile_name = QFileDialog.getOpenFileName(self, '测试模型', '' ,'file(*.h5)')
        if len(openfile_name[0]) == 0:
            return      
        path_name = openfile_name[0]
        self.test_model_line.setText(path_name)        
    
    def test_dbn_model(self):
        model_path = self.test_model_line.text()
        if len(model_path) == 0:
            QMessageBox.warning(self,'警告','未选择测试模型', QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)
            return 

        x_test_path = self.test_feature_line.text()
        y_test_path = self.test_label_line.text()
        if len(x_test_path) == 0 or len(y_test_path) == 0:
            QMessageBox.warning(self,'警告','未选择测试集', QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)
            return             
        self.model = DBNCLAS()
        self.model.load_test_data(x_test_path, y_test_path)
        self.model.load_model(model_path)
        acc, acc_mat, report = self.model.get_model_accuracy()
        self.acc_line.setText(str(acc))
        self.result_textedit.append('模型混淆矩阵: \n{}'.format(acc_mat))
        self.result_textedit.append('模型精度评估: \n{}'.format(report))     



class ModelTrainingThread(QThread):#线程类
    finish_signal = pyqtSignal(str)  
    def __init__(self):
        super(ModelTrainingThread, self).__init__()
        return 
        
    def set_training_model(self, model):
        self.model = model
        return 

    def run(self): #线程执行函数
        self.model.pretrain()  # 预训练
        history = self.model.fine_tune()  # 微调
        self.finish_signal.emit('训练完成')
       
        
class UI_DBNTL(QDialog):
    def __init__(self):
        super(UI_DBNTL, self).__init__()   
        self.resize(600, 600)
        self.setWindowTitle('模型迁移')
        self.setWindowIcon(QIcon('.\\logo\\安全.png'))
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
        

    def init_training_group(self):
        self.training_group_layout = QGridLayout()
        self.training_group.setLayout(self.training_group_layout)

        self.training_data_group = QGroupBox('训练数据')
        self.training_parameter_group = QGroupBox('微调参数设置')

        self.training_group_layout.addWidget(self.training_data_group, 0, 0)
        self.training_group_layout.addWidget(self.training_parameter_group, 1, 0) 

    def init_training_data_group(self):
        self.training_data_group_layout = QGridLayout()
        self.training_data_group.setLayout(self.training_data_group_layout)

        training_feature_label = QLabel('特征集')
        self.training_feature_line = QLineEdit()
        self.training_feature_line.setText('.\\训练数据\\x_train_example_tl.csv')
        training_feature_button = QPushButton('选择')
        training_feature_button.clicked.connect(self.choose_training_feature_data)

        training_label_label = QLabel('标签集')
        self.training_label_line = QLineEdit()
        self.training_label_line.setText('.\\训练数据\\y_train_example_tl.csv')
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
        self.source_model_line.setText('.\\代理模型\\model_source_example.h5')
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

    def init_train_prosess_group(self):
        self.test_result_group_layout = QGridLayout()
        self.train_prosess_group.setLayout(self.test_result_group_layout)
        self.result_textedit = QTextEdit()
        self.result_textedit.setPlainText('训练过程信息')
        self.test_result_group_layout.addWidget(self.result_textedit, 0, 0)         

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
            dialog = QMessageBox(QMessageBox.Warning, '警告', '未导入微调数据！！！')
            dialog.exec_()            
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

