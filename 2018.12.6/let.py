import pandas as pd
from os                                import environ, listdir, makedirs, getcwd
from os.path                           import isfile, join, exists
import tensorflow as tf
from readFiles import *  # 同一个文件下的另外一个自己写的函数库，这个错误可以忽略，已经加载成功

from sklearn.model_selection import GridSearchCV

from keras.models import Model, load_model
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Activation, Input, concatenate, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD, Adadelta, Adagrad, Adam
from keras.layers.normalization import BatchNormalization  # 批标准化（防止梯度爆炸或者梯度消失）
from keras.utils.vis_utils import plot_model
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.advanced_activations import ELU
import numpy as np



def genBatch(X, Y, batch_size, num_batches):
    while True:
        for batch_idx in range(num_batches):
            start_point = batch_idx*batch_size
            X_batch = []
            X_batch.append(X[start_point:start_point+batch_size])
            Y_batch   =    Y[start_point:start_point+batch_size]
            yield (X_batch, Y_batch)
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

#维度转化         恢复
Y_train=np.reshape(Y_train,(-1,1))
Y_test = np.reshape(Y_test,(-1,1))
X_train= np.reshape(X_train,(-1,11,1))
X_test= np.reshape(X_test,(-1,11,1))


#to_list
# Y_train=np.reshape(Y_train,(-1,1))
# Y_test = np.reshape(Y_test,(-1,1,))
# X_train= np.reshape(X_train,(-1,11,1))
# X_test= np.reshape(X_test,(-1,11,1))

rootpath = ''
outpath = '{}Results/{}/Results1D/{}Branches/'.format(rootpath, 'model', 1)
model_file = outpath + 'model.h5'

if not exists(outpath):
    makedirs(outpath)
num_depth = 1
init = 'he_normal'
momentum = 0.99
learn_rate=0.001
slope = 0.2
perc = 1.0
inputSize = 11
branches = []
inputs = []

# x_in = tf.placeholder(tf.float32, [None, 11,1], name='x-input')
# y_in = tf.placeholder(tf.float32, [None, 2], name='y-input')

i = 0
nLabels = 2
for i in range(num_depth):
	inputs.append(Input(shape=(inputSize,1), name='input' + str(i + 1)))
	branches.append(Conv1D(32, (15), padding='same', kernel_initializer=init)(inputs[i]))
	branches[i] = BatchNormalization(momentum=momentum)(branches[i])
	# branches[i] = LeakyReLU(slope)(branches[i])
	branches[i] = ELU(slope)(branches[i])
	branches[i] = MaxPooling1D(pool_size=2)(branches[i])
	branches[i] = Conv1D(64, (13), padding='same', kernel_initializer=init)(branches[i])
	branches[i] = BatchNormalization(momentum=momentum)(branches[i])
	branches[i] = LeakyReLU(slope)(branches[i])
	branches[i] = MaxPooling1D(pool_size=2)(branches[i])
	branches[i] = Conv1D(128, (11), padding='same', kernel_initializer=init)(branches[i])
	branches[i] = BatchNormalization(momentum=momentum)(branches[i])
	branches[i] = LeakyReLU(slope)(branches[i])
	branches[i] = MaxPooling1D(pool_size=2)(branches[i])
if num_depth > 1:
	x = concatenate(branches)
else:
	x = branches[i]
x = Flatten()(x)
x = Dense(256 * num_depth, kernel_initializer=init)(x)
x = LeakyReLU(slope)(x)
# x = ELU(slope)(x)
x = Dropout(perc)(x)
prediction = Dense(nLabels, activation='softmax', kernel_initializer=init)(x)
model = Model(inputs=inputs, outputs=prediction)
model.compile(loss='sparse_categorical_crossentropy',
              #optimizer=SGD(lr=learn_rate,momentum=momentum),
              optimizer=Adam(lr=learn_rate,epsilon=10e-8),
              metrics=['accuracy'])
model.save(model_file)
plot_model(model, to_file='model.png')

batch_size=10
num_batches=len(X_train) // batch_size
num_epoch = 1000
epoch=0
train_gen=genBatch(X_train,Y_train, batch_size, num_batches)

for epoch in range(epoch, num_epoch):
	model.fit_generator(generator=train_gen,steps_per_epoch=num_batches, initial_epoch=epoch,
				                    epochs=epoch+1
				                    # callbacks=[learning_rate_scheduler, h],
				                    # callbacks=[h],
				                    )

'''
#定义查询搜索参数网格
def create_model(model):
    # create model
    my_model = model
    return my_model
model = KerasClassifier(build_fn=create_model, nb_epoch=100, batch_size=10, verbose=0)
if __name__ == "__main__":
	learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
	momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
	param_grid = dict(learn_rate=learn_rate, momentum=momentum)
	grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)  # 网格搜索的交叉验证函数，n_jobs进程数
	grid_result = grid.fit(X_train, Y_train)
	print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
	for params, mean_score, scores in grid_result.grid_scores_:
		print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
'''


'''
#传统的训练模型
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_in,logits=prediction),name='cross_entropy')
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(y_in,1))#argmax返回一维张量中最大的值所在的位置
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(101):
		#batch_xs, batch_ys = mnist.train.next_batch(batch_size)
		sess.run(train_step, feed_dict={x_in: X_train, y_in: Y_train, keep_prob: 0.7})
		#batch_xs, batch_ys = mnist.test.next_batch(batch_size)
		if i % 10 == 0:
			test_acc = sess.run(accuracy, feed_dict={x_in: X_test, y_in: Y_test, keep_prob: 1.0})
			train_acc = sess.run(accuracy, feed_dict={x_in: X_train, y_in: Y_train, keep_prob: 1.0})
			print("Iter " + str(i) + ", Testing Accuracy= " + str(test_acc) + ", Training Accuracy= " + str(train_acc))
'''
'''
神经网路中的超参数主要包括:
	1. 学习率 ηη，2. 正则化参数 λλ，3. 神经网络的层数 LL，4. 每一个隐层中神经元的个数 jj，
	5. 学习的回合数EpochEpoch，6. 小批量数据 minibatchminibatch 的大小，7. 输出神经元的编码方式，
	8. 代价函数的选择，9. 权重初始化的方法，10. 神经元激活函数的种类，11.参加训练模型数据的规模.
'''