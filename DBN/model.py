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

from sklearn.neural_network import BernoulliRBM
from sklearn.neural_network import MLPRegressor 
from sklearn.metrics import explained_variance_score

import matplotlib.pyplot as plt



class DNN():
    def __init__(self):
        self.activation_function_nn = 'tanh'
        self.activation_function_output = 'linear'

        self.learning_rate_nn = 0.005
        self.batch_size_nn = 100
        self.n_epochs_nn = 400
        self.dropout_rate = 0.01
        self.decay_rate = 0
        self.model = Sequential()
        return 

    def get_hidden_layer_structure(self):
        hidden_layer_structure = self.hidden_layer_structure
        return hidden_layer_structure

    def set_hidden_layer_structure(self, hidden_layer_structure):
        self.hidden_layer_structure = hidden_layer_structure
        return

    def load_train_data(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        return 

    def load_test_data(self, x_test, y_test):
        self.x_test = x_test
        self.y_test = y_test
        return 

    def get_hidden_layer_activation_function(self):
        return self.activation_function_nn 

    def set_hidden_layer_activation_function(self, activation_function_nn):
        self.activation_function_nn = activation_function_nn
        return 

    def get_output_layer_activation_fuction(self):
        return self.activation_function_output 

    def set_output_layer_activation_fuction(self, activation_function_output):
        self.activation_function_output = activation_function_output
        return 

    def get_dropout_rate(self):
        return self.dropout_rate

    def set_dropout_rate(self, dropout_rate):
        self.dropout_rate = dropout_rate
        return

    def set_epochs_number(self, n_epochs_nn):
        self.n_epochs_nn = n_epochs_nn
        return 

    def build_model(self):
        hidden_layer_structure = self.get_hidden_layer_structure()

        for i in range(len(hidden_layer_structure)):
            if i == 0:
                self.model.add(Dense(units=hidden_layer_structure[i], activation=self.activation_function_nn, 
                    input_shape=(self.x_train.shape[1],), activity_regularizer=regularizers.l2(0.001)))

            elif i >= 1:
                self.model.add(Dense(units=hidden_layer_structure[i], activation=self.activation_function_nn, 
                    activity_regularizer=regularizers.l2(0.001)))
            else:
                pass
            self.model.add(Dropout(0.2))  # 丢弃神经元链接概率
        self.model.add(Dense(units=1, activation='identity'))
        return 

    def train(self):
        self.build_model()

        x_train = self.x_train
        x_test = self.x_test 
        y_train = self.y_train
        y_test = self.y_test

        adam = Adam(lr=0.0001, decay=0.0001)
        self.model.compile(loss='mse', optimizer=adam)

        history = self.model.fit(x_train, y_train, epochs=self.n_epochs_nn, batch_size=20, verbose=1, validation_data=(x_test, y_test))
 
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show() 

        eval = self.model.evaluate(x_test, y_test, verbose=0)
        print("Evaluation on test data: loss = {}".format(eval))
        return 

    def predict(self, x_predict):
        y_predict = self.model.predict(x_predict)
        return y_predict

class DBN():
    def __init__(self):
        self.learning_rate_rbm = 0.001
        self.batch_size_rbm = 100
        self.n_epochs_rbm = 50
        self.verbose_rbm = 1
        self.random_seed = 100

        self.activation_function_nn = 'relu'  

        self.learning_rate_nn = 0.001
        self.batch_size_nn = 50
        self.n_epochs_nn = 10  
        self.verbose_nn = 1
        self.decay_rate = 0
            
        self.hidden_layer_structure = []
        self.weight_rbm = []
        self.bias_rbm = []

        self.loss_output_file = ''
        self.loss_figure_output = False
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

    def save_model(self, name):
        self.model.save(name + '.h5')
        return 

    def load_model(self, name):
        self.model = load_model(name)
        return

    def set_parameter_optimization_result_file(self, parameter_optimization_result_file):
        self.parameter_optimization_result_file = parameter_optimization_result_file
        return

    def get_layer_weights(self, layer_num):
        weight, bias = self.model.get_layer(index=layer_num).get_weights()
        return weight, bias 
    
    def get_layer_output(self, layer_num, data):
        layer_model = Model(inputs=self.model.input, outputs=model.get_layer(index=layer_num).output)
        layer_output = layer_model.predict(data)
        return data 

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
                    activity_regularizer=regularizers.l2(0.01)))
            elif i >= 1:
                self.model.add(Dense(units=hidden_layer_structure[i], activation=self.activation_function_nn,
                    activity_regularizer=regularizers.l2(0.01)))
                
            else:
                pass
            layer = self.model.layers[i]
            layer.set_weights([self.weight_rbm[i], self.bias_rbm[i]])  
            
        self.model.add(Dense(units=1, activation=None)) 
        return 

    def fine_tune(self):
        x_train = self.x_train
        x_test = self.x_test 
        y_train = self.y_train
        y_test = self.y_test

        self.build_model()

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

        if len(self.loss_output_file) != 0:
            with open(self.loss_output_file, 'w', newline='') as f: 
                csv_write = csv.writer(f)
                csv_write.writerow(['train', 'test']) 
                for i in range(len(history.history['loss'])):
                    csv_write.writerow([history.history['loss'][i], history.history['val_loss'][i]])     
        return 

    def get_model_evaluate_result(self):
        loss = self.model.evaluate(self.x_test, self.y_test, verbose=0, batch_size=20)
        return loss

    def predict(self, x_predict):
        y_predict = self.model.predict(x_predict)
        return y_predict 

    
    def optimize_parameter(self):
        learning_rate = [0.1, 0.01, 0.001, 1e-4, 1e-5, 1e-6]
        decay_rate = [0.1, 0.01, 0.001, 1e-4, 1e-5, 1e-6]

        self.weight_rbm.clear()
        self.bias_rbm.clear()
        self.loss_output_file = ''
        with open(self.parameter_optimization_result_file, 'w', newline='') as f: 
            csv_write = csv.writer(f)
            csv_write.writerow(['nn_learning_rate', 'nn_decay_rate', 'loss']) 

        self.pretrain()
        n = 1
        for i in learning_rate:
            for j in decay_rate:
                self.set_nn_learning_rate(i)
                self.set_nn_decay_rate(j)
                self.set_train_process_loss_output_file('.\\parameter_optimization\\parameter_{}.csv'.format(n))
                
                self.fine_tune()
                loss = self.get_model_evaluate_result()
                with open(self.parameter_optimization_result_file, 'a+', newline='') as f: 
                    csv_write = csv.writer(f)
                    csv_write.writerow([i, j, loss])
                y_predict = self.model.predict(self.x_test)
                y_predict = y_predict.flatten()
                with open('.\\parameter_optimization\\predict_result_{}.csv'.format(n), 'a+', newline='') as f: 
                    csv_write = csv.writer(f)
                    csv_write.writerow(y_predict.tolist())
                n = n + 1
        return         


class DBN_transfer():
    def __init__(self):
        return
    def load_target_domain_train_data(self, x_train, y_train):
        '''
        加载目标域的样本数据
        '''
        self.x_train = x_train
        self.y_train = y_train
        return
    
    def load_source_domain_model(self, model):
        self.source_model = model 
        return 

    def set_transfer_method(self, transfer_method): 
        '''
        两种迁移方法, 微调整个网络还是调最后一层
        '''
        self.transfer_method = transfer_method
        return 

    def transfer_model(self):
        if self.transfer_method == 'regression':
            self.fine_tune_output_layer()
        elif self.transfer_method == 'all':
            self.fine_tune_whole_layer()
        else:
            pass 

    def fine_tune_whole_layer(self):
        return

    def freeze_layer(self, layer_num):
        return 
