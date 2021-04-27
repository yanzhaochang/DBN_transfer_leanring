import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

from sklearn.svm import SVR
from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

from model import DNN, DBN, TLM 


def train_DBN_model(): 
    '''
    使用安全裕度作为预测值，要修改学习率相关参数。
    '''
    x_train = pd.read_csv('..\\feature\\x_train_level_100.csv', header=None)
    x_train = x_train.values * 5   # 输入特征放缩到0-1之间, 相当于乘5
    
    y_train = pd.read_csv('..\\feature\\y_train_level_100.csv', header=None)
    y_train = y_train.values[:, 1]  # 最低频率

    x_test = pd.read_csv('..\\feature\\x_test_level_100.csv', header=None)
    x_test = x_test.values * 5

    y_test = pd.read_csv('..\\feature\\y_test_level_100.csv', header=None)
    y_test = y_test.values[:, 1]

    #x_val = 
    model = DBN()  # 初始化
    model.load_train_data(x_train, y_train)  # 加载训练数据和测试数据
    model.load_test_data(x_test, y_test)
    model.set_hidden_layer_structure([100, 30])  # 设置隐含层结构

    # 设置rbm训练参数
    model.set_rbm_epochs_number(50)  # RBM学习次数
    model.set_rbm_learning_rate(1e-4)  # RBM学习率
    model.set_rbm_batch_size(20)  # RBM批处理
    
    # 设置网络训练参数
    model.set_nn_learning_rate(0.01)  # 神经网络学习率
    model.set_nn_decay_rate(1e-4)  # 学习率下降参数
    model.set_hidden_layer_regularizer_l2(1e-4)   # 设置l2正则化系数
    model.set_nn_epochs_number(2000)  # 学习次数
    model.set_nn_batch_size(20)  # 学习批处理个数
    
    
    #model.set_train_process_loss_output_file('.\\train_process\\loss_1.csv')  # 设置训练过程保存
    model.set_train_process_loss_figure_output(True)  # 训练过程损失曲线
    model.set_predict_error_figure(True)  # 测试集上的绝对误差曲线
    # 模型训练
    model.pretrain()  # 预训练
    model.fine_tune()  # 微调
    
    acc = model.get_model_accuracy()  # 查看模型精度
    print('模型精确度: ', acc)  # 0.9958491599534041
    model.save_model('.\\models\\model_level_100')  # 保存模型
    
    '''
    # 以下为基础模型的参数寻优设置
    model.set_nn_epochs_number(1000)
    model.set_nn_batch_size(50)
    model.set_parameter_optimization_result_file('optimization_result.csv')
    model.optimize_parameter()
    '''
    return
'''
def check_DBN_model():
    model = DBN()
    model.load_model('.\\models\\model_level_100.h5')

    x_train = pd.read_csv('..\\feature\\x_train_level_100.csv', header=None)
    x_train = x_train.values 
    y_train = pd.read_csv('..\\feature\\y_train_level_100.csv', header=None)
    y_train = y_train.values[:, 0]  # 最低频率
    
    x_test = pd.read_csv('..\\feature\\x_test_level_100.csv', header=None)
    x_test = x_test.values 
    y_test = pd.read_csv('..\\feature\\y_test_level_100.csv', header=None)
    y_test = y_test.values[:, 0]

    y_predict = model.predict(x_test)
    print(y_predict)  
    return 

def train_target_domain_model(target_load_level):  # 训练迁移学习模型
    x_train = pd.read_csv('..\\feature\\x_train_level_{}.csv'.format(target_load_level), header=0)
    x_train = x_train.values * 5
    y_train = pd.read_csv('..\\feature\\y_train_level_{}.csv'.format(target_load_level), header=0)
    y_train = y_train.values[:, 1]  # 最低频率
    
    x_test = pd.read_csv('..\\feature\\x_test_level_{}.csv'.format(target_load_level), header=0)
    x_test = x_test.values * 5
    y_test = pd.read_csv('..\\feature\\y_test_level_{}.csv'.format(target_load_level), header=0)
    y_test = y_test.values[:, 1]

    model = TLM()
    model.load_model('.//models//model_level_100.h5')  # 加载源模型
    # 加载训练数据和测试数据, 测试数据集规模大于训练数据集
    model.load_train_data(x_train, y_train)
    model.load_test_data(x_test, y_test)
    
    model.set_predict_error_figure(True)
    acc = model.get_source_model_accuracy()  # 查看源模型在目标域下的精度
    print('源模型精确度: ', acc)  
    
    # 设置网络训练参数, 相对于初始模型训练，学习率和正则化参数都要设置低一些
    model.set_nn_learning_rate(0.00001)  # 学习率
    model.set_nn_decay_rate(1e-6)  # 学习率下降率
    model.set_train_process_loss_figure_output(True)  # 损失曲线
    model.set_hidden_layer_regularizer_l2(2e-6)   
    model.set_nn_epochs_number(1000)  # 学习代数
    model.set_nn_batch_size(20)  # 批数

    model.fine_tune()  # 全部网络参数进行微调

    acc = model.get_target_model_accuracy()
    print('目标模型的精度：', acc)
    model.save_model('.//models//model_level_{}'.format(target_load_level))  # 保存模型
    return

def train_MLP_model():  # 人工神经网络，只有一个隐含层
    load_level = 100
    x_train = pd.read_csv('..\\feature\\x_train_level_{}.csv'.format(load_level), header=0)
    x_train = x_train.values * 5
    y_train = pd.read_csv('..\\feature\\y_train_level_{}.csv'.format(load_level), header=0)
    y_train = y_train.values[:, 1]  # 最低频率
    y_train = y_train.astype(np.int16)

    x_test = pd.read_csv('..\\feature\\x_test_level_{}.csv'.format(load_level), header=0)
    x_test = x_test.values * 5
    y_test = pd.read_csv('..\\feature\\y_test_level_{}.csv'.format(load_level), header=0)
    y_test = y_test.values[:, 1]
    y_test = y_test.astype(np.int16)
    print(y_test.dtype)

    print('测试集上稳定样本数目:', np.sum(y_test))
    print('测试集总数目:', np.size(y_test))

    mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 5))
    mlp.fit(x_train, y_train)

    y_predict = mlp.predict(x_test)
    print(y_predict)
    score = mlp.score(x_test, y_test)
    print(score)
    report = classification_report(y_test, y_predict, target_names=['不稳定', '稳定'])
    print(report)

    confusion = confusion_matrix(y_test, y_predict)
    print(confusion)
    joblib.dump(mlp, './/models//mlp_{}.joblib'.format(load_level))  # 保存模型
    return

def train_SVR_model(target_load_level):
    x_train = pd.read_csv('..\\feature\\x_train_level_{}.csv'.format(target_load_level), header=0)
    x_train = x_train.values * 5
    y_train = pd.read_csv('..\\feature\\y_train_level_{}.csv'.format(target_load_level), header=0)
    y_train = y_train.values[:, 1]  # 最低频率
    
    x_test = pd.read_csv('..\\feature\\x_test_level_{}.csv'.format(target_load_level), header=0)
    x_test = x_test.values * 5
    y_test = pd.read_csv('..\\feature\\y_test_level_{}.csv'.format(target_load_level), header=0)
    y_test = y_test.values[:, 1]

    model = SVR(kernel='rbf', gamma='auto', C=100, tol=1e-4, epsilon=0.001, verbose=True, max_iter=1000)

    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)
    y_predict = y_predict.reshape(-1)
    acc = 1 - np.sum(np.abs(y_predict-y_test)) / (np.size(y_predict) * (np.max(y_train) - np.min(y_train)))  
    print('模型评分', acc) 
    #score = model.score(x_test, y_test)
    #print('SVR模型的评分:', score)
    

    # 绘制绝对误差的图形
    plt.figure(1)
    x = np.array(range(1, np.size(y_test)+1))
    y = abs(y_predict - y_test)    
    plt.plot(x, y)
    plt.show()
    return
'''
def train_SVM_model():
    '''利用SVM训练一个二分类模型
    '''
    load_level = 100
    x_train = pd.read_csv('..\\feature\\x_train_level_{}.csv'.format(load_level), header=0)
    x_train = x_train.values * 5
    y_train = pd.read_csv('..\\feature\\y_train_level_{}.csv'.format(load_level), header=0)
    y_train = y_train.values[:, 1]  # 最低频率
    y_train = y_train.astype(np.int16)

    x_test = pd.read_csv('..\\feature\\x_test_level_{}.csv'.format(load_level), header=0)
    x_test = x_test.values * 5
    y_test = pd.read_csv('..\\feature\\y_test_level_{}.csv'.format(load_level), header=0)
    y_test = y_test.values[:, 1]
    y_test = y_test.astype(np.int16)
    print(y_test.dtype)

    print('测试集上稳定样本数目:', np.sum(y_test))
    print('测试集总数目:', np.size(y_test))

    svm = SVC(C=20.0, kernel='rbf', gamma='auto')
    svm.fit(x_train, y_train)

    y_predict = svm.predict(x_test)
    print(y_predict)
    score = svm.score(x_test, y_test)
    print(score)
    report = classification_report(y_test, y_predict, target_names=['不稳定', '稳定'])
    print(report)
    
    confusion = confusion_matrix(y_test, y_predict)
    print(confusion)
    print('支持向量:', svm.support_vectors_)
    print('支持向量的索引:', svm.support_)
    print('每一类支持向量的个数:', svm.n_support_)
    ss = decision_function(y_test)
    print(ss)
    joblib.dump(svm, 'svm.pkl')  # 保存模型    
    return

if __name__=='__main__':
    #train_DBN_model()
    #train_MLP_model()
    train_SVM_model()
    #check_DBN_model()

    #target_load_level = 105
    #train_SVR_model(target_load_level)
    #train_DNN_model()
    
    #train_target_domain_model(target_load_level)