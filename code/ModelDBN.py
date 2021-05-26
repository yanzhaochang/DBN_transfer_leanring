# 训练分类模型
import pandas as pd 
import numpy as np 
import csv 

from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout
from keras import regularizers, backend  # 正则化
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping

from sklearn.neural_network import BernoulliRBM
from sklearn.metrics import confusion_matrix, classification_report



class DBNCLAS():  # 这是一个多输入单输出模型
    def __init__(self, hidden_layer_structure=[50, 10]):
        self.learning_rate_rbm = 0.001
        self.batch_size_rbm = 100
        self.n_epochs_rbm = 50
        self.verbose_rbm = 1
        self.random_seed = 100

        self.activation_function_nn = 'relu'  
        self.regularizer_l2_nn = 0.001
        
        self.learning_rate_nn = 0.001
        self.batch_size_nn = 50
        self.n_epochs_nn = 10  
        self.verbose_nn = 1
        self.decay_rate = 0

        self.hidden_layer_structure = hidden_layer_structure

    def load_train_data(self, x_train_path, y_train_path, validation_split):
        x_train = pd.read_csv(x_train_path, header=0, encoding='gbk', engine='python')
        x_train = x_train.values * 5

        y_train = pd.read_csv(y_train_path, header=0, encoding='gbk', engine='python')
        y_train = y_train.values[:, 2]  
        y_train = y_train.astype(np.int16)   

        self.x_train = x_train
        self.y_train = y_train
        self.validation_split = validation_split
        return 

    def load_test_data(self, x_test_path, y_test_path):
        x_test = pd.read_csv(x_test_path, header=0, encoding='gbk', engine='python')
        x_test = x_test.values * 5

        y_test = pd.read_csv(y_test_path, header=0, encoding='gbk', engine='python')
        y_test = y_test.values[:, 2]  
        y_test = y_test.astype(np.int16)  

        self.x_test = x_test
        self.y_test = y_test
        return 

    def get_hidden_layer_structure(self):
        hidden_layer_structure = self.hidden_layer_structure
        return hidden_layer_structure

    def set_hidden_layer_structure(self, hidden_layer_structure):
        self.hidden_layer_structure = hidden_layer_structure
        return 

    def set_rbm_learning_rate(self, learning_rate_rbm):
        self.learning_rate_rbm = learning_rate_rbm
        return 
    
    def set_rbm_epochs_number(self, n_epochs_rbm):
        self.n_epochs_rbm = n_epochs_rbm
        return

    def set_rbm_batch_size(self, batch_size_rbm):
        self.batch_size_rbm = batch_size_rbm
        return

    def set_nn_epochs_number(self, n_epochs_nn):
        self.n_epochs_nn = n_epochs_nn
        return

    def set_nn_learning_rate(self, learning_rate_nn):
        self.learning_rate_nn = learning_rate_nn
        return

    def set_nn_decay_rate(self, decay_rate):
        self.decay_rate = decay_rate
        return

    def set_nn_batch_size(self, batch_size_nn):
        self.batch_size_nn = batch_size_nn
        return

    def get_model_accuracy(self):
        loss, acc = self.model.evaluate(self.x_test, self.y_test)
        y_pred = self.model.predict_classes(self.x_test)
        acc_mat = confusion_matrix(self.y_test, y_pred)  
        acc_report = classification_report(self.y_test, y_pred, target_names=['失稳', '稳定'])
        return acc, acc_mat, acc_report

    def save_model(self, name):
        self.model.save(name)
        return 

    def load_model(self, name):
        backend.clear_session()
        self.model = load_model(name)
        return

    def get_model_layer(self, layer_num):
        layer = self.model.get_layer(index=layer_num)
        return layer 

    def get_layer_weights(self, layer_num):
        layer = self.get_model_layer(layer_num)
        if layer.use_bias is True:
            weight, bias = layer.get_weights()
        else:
            weight = layer.get_weights()
            bias = 0
        return weight, bias

    def get_layer_output(self, layer_num, data):
        layer = self.get_model_layer(layer_num)
        layer_model = Model(inputs=self.model.input, outputs=layer.output)
        layer_output = layer_model.predict(data)
        return layer_output 

    def set_hidden_layer_regularizer_l2(self, regularizer_l2_nn):
        self.regularizer_l2_nn = regularizer_l2_nn
        return

    def pretrain(self):
        self.weight_rbm = []
        self.bias_rbm = []

        x_train = self.x_train
        y_train = self.y_train

        hidden_layer_structure = self.get_hidden_layer_structure()

        input_layer = x_train
        for i in range(len(hidden_layer_structure)):
            rbm = BernoulliRBM(n_components=hidden_layer_structure[i],
                               learning_rate=self.learning_rate_rbm,
                               batch_size=self.batch_size_rbm,
                               n_iter=self.n_epochs_rbm,
                               verbose=1,
                               random_state=self.random_seed)
            rbm.fit(input_layer)
            self.weight_rbm.append(rbm.components_.T)
            self.bias_rbm.append(rbm.intercept_hidden_)  
            input_layer = rbm.transform(input_layer)
        return        

    def build_model(self):
        backend.clear_session()
        self.model = Sequential()
        hidden_layer_structure = self.get_hidden_layer_structure()
        input_dim = self.x_train.shape[1]

        for i in range(0, len(hidden_layer_structure)):
            if i == 0:
                self.model.add(Dense(units=hidden_layer_structure[i], activation=self.activation_function_nn, input_dim=input_dim, 
                    activity_regularizer=regularizers.l2(self.regularizer_l2_nn)))
            elif i >= 1:
                self.model.add(Dense(units=hidden_layer_structure[i], activation=self.activation_function_nn,  
                    activity_regularizer=regularizers.l2(self.regularizer_l2_nn)))              
            else:
                pass
            layer = self.model.layers[i]
            layer.set_weights([self.weight_rbm[i], self.bias_rbm[i]])  

        self.model.add(Dense(units=1, activation='sigmoid', use_bias=True)) 
        return 

    def fine_tune(self):
        x_train = self.x_train
        y_train = self.y_train

        self.build_model()
        callback = EarlyStopping(monitor='val_acc', min_delta=1e-6, patience=200, verbose=1, mode='max', restore_best_weights=True)
        
        adam = Adam(lr=self.learning_rate_nn, decay=self.decay_rate)
        self.model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
        history = self.model.fit(x_train, y_train, validation_split=self.validation_split, 
            batch_size=self.batch_size_nn, epochs=self.n_epochs_nn, verbose=1, callbacks=[callback])


        return history

    def predict(self, x_predict):
        y_predict = self.model.predict(x_predict)
        return y_predict



class DBNTL():  # 定义一个迁移学习模型对象
    def __init__(self):
        self.weight_source_model = []
        self.bias_source_model = []

        self.activation_function_nn = 'relu'  
        self.regularizer_l2_nn = 0.001
        
        self.learning_rate_nn = 0.001
        self.batch_size_nn = 50
        self.n_epochs_nn = 10  
        self.decay_rate = 0
        return

    def load_train_data(self, x_train_path, y_train_path, val_split):
        '''加载目标域的样本数据'''
        x_train = pd.read_csv(x_train_path, header=0, encoding='gbk', engine='python')
        x_train = x_train.values * 5

        y_train = pd.read_csv(y_train_path, header=0, encoding='gbk', engine='python')
        y_train = y_train.values[:, 2]  
        y_train = y_train.astype(np.int16)
        
        print('训练样本中稳定样本比例{}'.format(np.sum(y_train)))
        self.x_train = x_train
        self.y_train = y_train
        self.val_split = val_split
        return

    def load_test_data(self, x_test_path, y_test_path):
        '''加载目标域的样本数据'''
        x_test = pd.read_csv(x_test_path, header=0, encoding='gbk', engine='python')
        x_test = x_test.values * 5

        y_test = pd.read_csv(y_test_path, header=0, encoding='gbk', engine='python')
        y_test = y_test.values[:, 2]
        y_test = y_test.astype(np.int16)       
        print('测试样本中稳定样本比例{}'.format(np.sum(y_test)))
        self.x_test = x_test
        self.y_test = y_test
        return  

    def load_model(self, name):
        self.source_model = load_model(name)
        return 

    def save_model(self, name):
        self.target_model.save(name)
        return 

    def set_nn_epochs_number(self, n_epochs_nn):
        self.n_epochs_nn = n_epochs_nn
        return

    def set_nn_learning_rate(self, learning_rate_nn):
        self.learning_rate_nn = learning_rate_nn
        return

    def set_nn_decay_rate(self, decay_rate):
        self.decay_rate = decay_rate
        return

    def set_nn_batch_size(self, batch_size_nn):
        self.batch_size_nn = batch_size_nn
        return

    def get_model_layer(self, layer_num):
        layer = self.target_model.get_layer(index=layer_num)
        return layer 

    def get_layer_weights(self, layer_num):
        layer = self.get_model_layer(layer_num)
        if layer.use_bias is True:
            weight, bias = layer.get_weights()
        else:
            weight = layer.get_weights()
            bias = 0
        return weight, bias

    def get_layer_output(self, layer_num):
        layer = self.get_model_layer(layer_num)
        layer_model = Model(inputs=self.target_model.input, outputs=layer.output)
        layer_output = layer_model.predict(self.x_train)
        return layer_output 


    def set_hidden_layer_regularizer_l2(self, regularizer_l2_nn):
        self.regularizer_l2_nn = regularizer_l2_nn
        return

    def set_target_model_layer_train_ability(self, layer_num, ability):
        '''
        设置目标模型某层的可训练性
        '''
        layer = self.target_model.get_layer(index=layer_num) 
        layer.trainable = ability
        return 

    def build_target_model(self):
        # 获取源模型的结构、权重
        layer_num = len(self.source_model.layers)
        self.target_model = Sequential()
        hidden_layer_structure = []  # 隐藏层结构
        for i in range(layer_num):
            layer = self.source_model.get_layer(index=i)
            layer_config = layer.get_config()
            units = layer_config['units']
            hidden_layer_structure.append(units)
            weight, bias = layer.get_weights()

            self.weight_source_model.append(weight)
            self.bias_source_model.append(bias) 
                    
        del hidden_layer_structure[-1]  # 删除结构中最后一个

        input_dim = self.x_train.shape[1]

        for i in range(0, len(hidden_layer_structure)):
            if i == 0:
                self.target_model.add(Dense(units=hidden_layer_structure[i], activation=self.activation_function_nn, input_dim=input_dim, 
                    activity_regularizer=regularizers.l2(self.regularizer_l2_nn)))
                
            elif i >= 1:
                self.target_model.add(Dense(units=hidden_layer_structure[i], activation=self.activation_function_nn,  
                    activity_regularizer=regularizers.l2(self.regularizer_l2_nn)))
                
            else:
                pass
            layer = self.target_model.layers[i]
            layer.set_weights([self.weight_source_model[i], self.bias_source_model[i]])  
            
        self.target_model.add(Dense(units=1, activation='sigmoid', use_bias=True))  
        layer = self.target_model.layers[-1]
        layer.set_weights([self.weight_source_model[-1], self.bias_source_model[-1]]) 
        return

    def fine_tune(self):
        callback = EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=50, verbose=1, mode='max', restore_best_weights=True)
        
        adam = Adam(lr=self.learning_rate_nn, decay=self.decay_rate)
        self.target_model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
        
        history = self.target_model.fit(self.x_train, self.y_train, validation_split=0.2, 
            batch_size=self.batch_size_nn, epochs=self.n_epochs_nn, verbose=1, callbacks=[callback])
        
        return history
        
    