import numpy as np

import pandas as pd 

from model import DNN, DBN 


def train_DBN_model(): 
    x_train = pd.read_csv('..\\feature\\x_train.csv', header=None)
    x_train = x_train.values 
    y_train = pd.read_csv('..\\feature\\y_train.csv', header=None)
    y_train = y_train.values[:, 0]
    x_test = pd.read_csv('..\\feature\\x_test.csv', header=None)
    x_test = x_test.values 
    y_test = pd.read_csv('..\\feature\\y_test.csv', header=None)
    y_test = y_test.values[:, 0]

    model = DBN()
    
    # 加载训练数据和测试数据
    model.load_train_data(x_train, y_train)
    model.load_test_data(x_test, y_test)

    # 设置隐含层结构
    model.set_hidden_layer_structure([50, 30, 10])

    # 设置rbm训练参数
    model.set_rbm_epochs_number(20)
    model.set_rbm_learning_rate(0.0001)
    model.set_rbm_batch_size(20)
    '''
    # 设置网络训练参数
    model.set_nn_learning_rate(0.001)
    model.set_nn_decay_rate(0.00001)
    '''
    model.set_nn_epochs_number(2000)
    model.set_nn_batch_size(20)
    '''
    # 设置训练过程结果保存
    model.set_train_process_loss_output_file('.\\train_process\\loss_2.csv')
    model.set_train_process_loss_figure_output(True)
    
    # 模型训练
    model.pretrain()
    model.fine_tune()
    loss = model.get_model_evaluate_result()
    print('测试集上的损失: {}'.format(loss))
    '''
    # 参数寻优
    model.set_parameter_optimization_result_file('optimization_result.csv')
    model.optimize_parameter()
    '''
    # 保存模型
    model.save_model('.\\models\\model_2')

    y_predict = model.predict(x_test)
    for i in range(0, 20):
        print(y_predict[i, 0], y_test[i])
    '''   
    return

def check_DBN_model():
    model = DBN()
    model.load_model('.\\models\\model_1.h5')
    weight, bias = model.get_layer_weights(3)
    print('权重矩阵：\n', weight)
    print('偏置矩阵:\n', bias)
    return 

def transfer_DBN_model():
    
    X_train_new = pd.read_csv('..\\level110\\features\\X_train.csv', header=None)
    X_train_new = X_train_new.values 

    y_train_new = pd.read_csv('..\\level110\\features\\y_train.csv', header=None)
    y_train_new = y_train_new.values[0]

    X_test_new = pd.read_csv('..\\level110\\features\\X_test.csv', header=None)
    X_test_new = X_test_new.values

    y_test_new = pd.read_csv('..\\level110\\features\\y_test.csv', header=None)
    y_test_new = y_test_new.values[0]

    test_data = {'x_test': X_train_new, 'y_test': y_train_new}
    train_data = {'x_train': X_test_new, 'y_train': y_test_new}
    print(X_train_new.shape)
    model = DBN_transfer()
    model.load_model(DBN_model.h5)
    model.retrain_classification_layer(train_data, test_data)
    
    return

def train_DNN_model():
    x_train = pd.read_csv('..\\feature\\x_train.csv', header=None)
    x_train = x_train.values 
    y_train = pd.read_csv('..\\feature\\y_train.csv', header=None)
    y_train = y_train.values[:, 0].reshape((-1, 1))
    x_test = pd.read_csv('..\\feature\\x_test.csv', header=None)
    x_test = x_test.values 
    y_test = pd.read_csv('..\\feature\\y_test.csv', header=None)
    y_test = y_test.values[:, 0].reshape((-1, 1))

    model = DNN()
    model.load_train_data(x_train, y_train)
    model.load_test_data(x_test, y_test)
    model.set_epochs_number(2000)
    model.set_hidden_layer_structure([50, 30, 10])
    model.train()
    y_predict = model.predict(x_test)
    for i in range(0, 20):
        print(y_predict[i, 0], y_test[i])
    return



if __name__=='__main__':
    train_DBN_model()
    #check_DBN_model()
