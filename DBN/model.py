'''
深度学习模型, 多输入单输出
'''
import os
import csv 
import numpy as np 

from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout
from keras import regularizers  # 正则化
from keras.optimizers import SGD, Adam
import keras.backend as K
from keras.callbacks import LearningRateScheduler,ModelCheckpoint, TensorBoard
from keras.losses import mse

from sklearn.neural_network import BernoulliRBM
from sklearn.neural_network import MLPRegressor 
from sklearn.metrics import explained_variance_score

import matplotlib.pyplot as plt


class DNN():
    def __init__(self):
        self.activation_function_nn = 'relu'  

        self.learning_rate_nn = 0.001
        self.batch_size_nn = 50
        self.n_epochs_nn = 10  
        self.verbose_nn = 1
        self.decay_rate = 0
            
        self.hidden_layer_structure = []
        self.regularizer_l2_nn = 0.001

        self.loss_output_file = ''
        self.loss_figure_output = False
        self.model = Sequential()
        return 

    def load_train_data(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        return 

    def load_test_data(self, x_test, y_test):
        self.x_test = x_test
        self.y_test = y_test
        return 

    def get_hidden_layer_structure(self):
        hidden_layer_structure = self.hidden_layer_structure
        return hidden_layer_structure

    def set_hidden_layer_structure(self, hidden_layer_structure):
        self.hidden_layer_structure = hidden_layer_structure
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

    def set_train_process_loss_output_file(self, loss_output_file):
        self.loss_output_file = loss_output_file
        return

    def set_train_process_loss_figure_output(self, loss_figure_output):
        self.loss_figure_output = loss_figure_output
        return 

    def get_test_sample_error(self):
        y_predict = self.predict(self.x_test)
        loss = mse(self.y_test, y_predict)
        return loss 

    def save_model(self, name):
        self.model.save(name + '.h5')
        return 

    def load_model(self, name):
        self.model = load_model(name)
        return

    def set_parameter_optimization_result_file(self, parameter_optimization_result_file):
        self.parameter_optimization_result_file = parameter_optimization_result_file
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

    def build_model(self):
        hidden_layer_structure = self.get_hidden_layer_structure()

        for i in range(len(hidden_layer_structure)):
            if i == 0:
                self.model.add(Dense(units=hidden_layer_structure[i], activation=self.activation_function_nn, input_shape=(self.x_train.shape[1],), 
                    activity_regularizer=regularizers.l2(self.regularizer_l2_nn)))

            elif i >= 1:
                self.model.add(Dense(units=hidden_layer_structure[i], activation=self.activation_function_nn, 
                    activity_regularizer=regularizers.l2(self.regularizer_l2_nn)))
            else:
                pass
            #self.model.add(Dropout(0.2))  # 丢弃神经元链接概率
        self.model.add(Dense(units=1, activation=None))
        return 

    def train(self):
        self.build_model()

        x_train = self.x_train
        x_test = self.x_test 
        y_train = self.y_train
        y_test = self.y_test

        adam = Adam(lr=self.learning_rate_nn, decay=self.decay_rate)
        self.model.compile(loss='mse', optimizer=adam)
    
        history = self.model.fit(x_train, y_train, batch_size=self.batch_size_nn, epochs=self.n_epochs_nn, verbose=1, validation_data=(x_test, y_test))
        
        if self.loss_figure_output is True:
            plt.figure(1)        
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model training loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show() 

    def get_model_evaluate_result(self):
        loss = self.model.evaluate(self.x_test, self.y_test, verbose=0, batch_size=20)
        return loss

    def predict(self, x_predict):
        y_predict = self.model.predict(x_predict)
        return y_predict 

class DBN():  # 这是一个多输入单输出模型
    def __init__(self):
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
            
        self.hidden_layer_structure = [1]
        self.weight_rbm = []
        self.bias_rbm = []

        self.loss_output_file = ''
        self.loss_figure_output = False
        self.predict_error_figure = False
        return 

    def load_train_data(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        return 

    def load_test_data(self, x_test, y_test):
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

    def set_train_process_loss_output_file(self, loss_output_file):
        self.loss_output_file = loss_output_file
        return

    def set_train_process_loss_figure_output(self, loss_figure_output):
        self.loss_figure_output = loss_figure_output
        return 

    def set_predict_error_figure(self, value):  # 查看测试集误差曲线
        self.predict_error_figure = value
        return

    def get_model_accuracy(self):
        y_predict = self.model.predict(self.x_test)
        y_predict = y_predict.reshape(-1)
        acc = 1 - np.sum(np.abs(y_predict-self.y_test)) / (np.size(y_predict) * (np.max(self.y_train) - np.min(self.y_train)))   
        if self.predict_error_figure is True:
            # 绘制绝对误差的图形
            plt.figure(1)
            x = np.array(range(1, np.size(self.y_test)+1))
            y = abs(y_predict - self.y_test)    
            plt.scatter(x, y)
            plt.show()            
        return acc

    def save_model(self, name):
        self.model.save(name + '.h5')
        return 

    def load_model(self, name):
        self.model = load_model(name)
        return

    def set_parameter_optimization_result_file(self, parameter_optimization_result_file):
        self.parameter_optimization_result_file = parameter_optimization_result_file
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

        self.model.add(Dense(units=1, activation=None, use_bias=True)) 
        return 

    def fine_tune(self):
        x_train = self.x_train
        y_train = self.y_train

        self.build_model()

        adam = Adam(lr=self.learning_rate_nn, decay=self.decay_rate)
        self.model.compile(loss='mse', optimizer=adam)
        
        history = self.model.fit(x_train, y_train, batch_size=self.batch_size_nn, epochs=self.n_epochs_nn, verbose=1)
        
        if self.loss_figure_output is True:
            plt.figure(1)        
            plt.plot(history.history['loss'])
            plt.title('model training loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train'], loc='upper left')
            plt.show() 

        if len(self.loss_output_file) != 0:
            with open(self.loss_output_file, 'w', newline='') as f: 
                csv_write = csv.writer(f)
                csv_write.writerow(['train']) 
                for i in range(len(history.history['loss'])):
                    csv_write.writerow([history.history['loss'][i]])     
        return 

    def get_model_evaluate_result(self):
        y_test = self.y_test 
        y_train = self.y_train

        y_predict = self.model.predict(self.x_test)
        y_predict = y_predict.reshape(-1)
        loss = self.model.evaluate(self.x_test, self.y_test, verbose=0, batch_size=20)
        acc = 1 - np.sum(np.abs(y_predict-y_test)) / (np.size(y_predict) * (np.max(y_train) - np.min(y_train)))          
        return loss, acc

    def predict(self, x_predict):
        y_predict = self.model.predict(x_predict)
        return y_predict 

    
    def optimize_parameter(self):
        learning_rate = [0.1, 0.01, 0.001, 1e-4]
        decay_rate = [0.001, 1e-4, 1e-5, 1e-6]
        regularizer_l2 = [1e-3, 1e-4, 1e-5, 1e-6]

        self.loss_output_file = ''
        with open(self.parameter_optimization_result_file, 'w', newline='') as f: 
            csv_write = csv.writer(f)
            csv_write.writerow(['nn_learning_rate', 'nn_decay_rate', 'regularizer_l2', 'loss', 'acc']) 

        n = 1
        for i in learning_rate:
            for j in decay_rate:
                for k in regularizer_l2:
                    self.weight_rbm.clear()
                    self.bias_rbm.clear()
                    self.set_nn_learning_rate(i)
                    self.set_nn_decay_rate(j)
                    self.set_hidden_layer_regularizer_l2(k)
                    self.set_train_process_loss_output_file('.\\parameter_optimization\\parameter_{}.csv'.format(n))
                    
                    self.pretrain()
                    self.fine_tune()
                    loss, acc = self.get_model_evaluate_result()
                    with open(self.parameter_optimization_result_file, 'a+', newline='') as f: 
                        csv_write = csv.writer(f)
                        csv_write.writerow([i, j, k,  loss, acc])
        
                    n = n + 1
        return         


class TLM():  # 定义一个迁移学习模型对象
    def __init__(self):

        self.weight_source_model = []
        self.bias_source_model = []

        self.activation_function_nn = 'relu'  
        self.regularizer_l2_nn = 0.001
        
        self.learning_rate_nn = 0.001
        self.batch_size_nn = 50
        self.n_epochs_nn = 10  
        self.verbose_nn = 1
        self.decay_rate = 0

        self.loss_figure_output = False 
        self.loss_output_file = ''
        self.predict_error_figure = False
        return

    def load_train_data(self, x_train, y_train):
        '''
        加载目标域的样本数据
        '''
        self.x_train = x_train
        self.y_train = y_train
        self.train_num = np.size(y_train)
        return

    def load_test_data(self, x_test, y_test):
        '''
        加载目标域的样本数据
        '''
        self.x_test = x_test
        self.y_test = y_test
        self.test_num = np.size(y_test)
        return  

    def set_train_sample_number(self, train_num):
        self.train_num = train_num
        return 

    def set_test_sample_number(self, test_num):
        self.test_num = test_num
        return 

    def load_model(self, name):
        self.source_model = load_model(name)
        return 

    def save_model(self, name):
        self.target_model.save(name + '.h5')
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

    def set_train_process_loss_output_file(self, loss_output_file):
        self.loss_output_file = loss_output_file
        return

    def set_train_process_loss_figure_output(self, loss_figure_output):
        self.loss_figure_output = loss_figure_output
        return 

    def set_predict_error_figure(self, value):  # 查看测试集误差曲线
        self.predict_error_figure = value
        return

    def get_source_model_accuracy(self):  # 获取源模型在测试集上的精度
        y_predict = self.source_model.predict(self.x_test)
        y_predict = y_predict.reshape(-1)
        acc = 1 - np.sum(np.abs(y_predict-self.y_test)) / (np.size(y_predict) * (np.max(self.y_train) - np.min(self.y_train)))   
        if self.predict_error_figure is True:
            # 绘制绝对误差的图形
            plt.figure(1)
            x = np.array(range(1, np.size(self.y_test)+1))
            y = abs(y_predict - self.y_test)    
            plt.scatter(x, y)
            plt.show()    
        return acc 

    def set_parameter_optimization_result_file(self, parameter_optimization_result_file):
        self.parameter_optimization_result_file = parameter_optimization_result_file
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

    def get_layer_output(self, layer_num, data):
        layer = self.get_model_layer(layer_num)
        layer_model = Model(inputs=self.model.input, outputs=layer.output)
        layer_output = layer_model.predict(data)
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
            
        self.target_model.add(Dense(units=1, activation=None, use_bias=True))  
        layer = self.target_model.layers[-1]
        layer.set_weights([self.weight_source_model[-1], self.bias_source_model[-1]]) 
        return

    def fine_tune(self):
        self.build_target_model()

        x_train = self.x_train[0:self.train_num, :]
        x_test = self.x_test[0:self.test_num, :] 
        y_train = self.y_train[0:self.train_num]
        y_test = self.y_test[0:self.test_num]

        adam = Adam(lr=self.learning_rate_nn, decay=self.decay_rate)
        self.target_model.compile(loss='mse', optimizer=adam)
    
        history = self.target_model.fit(x_train, y_train, batch_size=self.batch_size_nn, epochs=self.n_epochs_nn, verbose=1)
        
        if self.loss_figure_output is True:
            plt.figure(1)        
            plt.plot(history.history['loss'])
            plt.title('model training loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train'], loc='upper left')
            plt.show() 

        if len(self.loss_output_file) != 0:
            with open(self.loss_output_file, 'w', newline='') as f: 
                csv_write = csv.writer(f)
                csv_write.writerow(['train', 'test']) 
                for i in range(len(history.history['loss'])):
                    csv_write.writerow([history.history['loss'][i], history.history['val_loss'][i]])     
        return 

    def get_target_model_accuracy(self):  # 目标模型精度
        x_train = self.x_train[0:self.train_num, :]
        x_test = self.x_test[0:self.test_num, :] 
        y_train = self.y_train[0:self.train_num]
        y_test = self.y_test[0:self.test_num]

        y_predict = self.target_model.predict(x_test)
        y_predict = y_predict.reshape(-1)
        acc = 1 - np.sum(np.abs(y_predict-y_test)) / (np.size(y_predict) * (np.max(y_train) - np.min(y_train)))   
        if self.predict_error_figure is True:  # 绘制绝对误差的图形
            plt.figure(1)
            x = np.array(range(1, np.size(y_test)+1))
            y = abs(y_predict - y_test)    
            plt.scatter(x, y)
            plt.ylim(0.0, 0.01)
            plt.show()   
        #for i in range(np.size(y_predict)):
        #    print(y_predict[i], y_test[i])      
        return acc

    def predict(self, x_predict):
        y_predict = self.target_model.predict(x_predict)
        return y_predict