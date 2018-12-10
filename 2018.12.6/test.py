import numpy as np
import tensorflow as tf
import random
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.models import load_model

from readFiles import *


my_model = load_model('Results/model/Results1D/1Branches/model.h5')

fileName = 'features.csv'
x_list = []  # 数据样本特征向量
y_list = []  # 样本标签
x_list, y_list = read_data(fileName)  # 忽略

print(x_list[:5])
print(y_list[:5])
X_train, X_test, Y_train, Y_test = split_data(x_list, y_list)

#one_hot 格式化标签
CLASS=2

lables1=tf.constant(Y_train)
lables_train= tf.one_hot(lables1,CLASS,1,0)
lables2=tf.constant(Y_test)
lables_test = tf.one_hot(lables2,CLASS,1,0)


#特征向量（list->array）  不知对与不对
# X_train = np.array(X_train)
# X_test = np.array(X_test)
# Y_train = np.array(lables_train)
# Y_test = np.array(lables_test)

#维度转化
Y_train=np.reshape(Y_train,(-1,1))
Y_test = np.reshape(Y_test,(-1,1))
X_train= np.reshape(X_train,(-1,11,1))
X_test= np.reshape(X_test,(-1,11,1))
# print(Y_test)
pred = my_model.predict(X_test)
print(pred)
print(Y_test)
score = my_model.evaluate(X_test, Y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# print('Label of testing sample', np.argmax(Y_test[:]))
print('Label of testing sample', Y_test[5:6])   #第五个测试数据的真实标签
print('Output of the softmax layer\n', pred[5])  #第五个测试数据输入模型后计算的输出结果
print('Network prediction:\n', np.argmax([pred[5]])) #根据第五个测试数据输出结果预测的标签值

