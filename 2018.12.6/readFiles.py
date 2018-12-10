import csv
from sklearn.model_selection import train_test_split

#读取文件
def read_data(test_data='features.csv', n=0, label=1):
	'''
	加载数据的功能
	n:特征数据起始位n代表每一条样本中特征数据的起始位，如果csv文件每个样本行第一个数据单元为样本的id，不参与计算，
	那么n=1，我这里用的是n=0
	label：是否是监督样本数据

	'''
	csv_reader = csv.reader(open(test_data))
	data_list = []
	for one_line in csv_reader:
		data_list.append(one_line)
	x_list = []
	y_list = []
	for one_line in data_list[1:]:
		if label == 1:
			y_list.append(int(one_line[-1]))  # 标志位
			one_list = [float(o) for o in one_line[n:-1]]
			x_list.append(one_list)
		else:
			one_list = [float(o) for o in one_line[n:]]
			x_list.append(one_list)
	return x_list, y_list

#分离训练和测试样本
def split_data(x_list, y_list, ratio=0.30):
    '''
    按照指定的比例，划分样本数据集
    ratio: 测试数据的比率
    '''
    X_train, X_test, y_train, y_test = train_test_split(x_list, y_list, test_size=ratio, random_state=50)
    print('--------------------------------split_data shape-----------------------------------')
    print(len(X_train), len(y_train))
    print(len(X_test), len(y_test))
    return X_train, X_test, y_train, y_test


def updatePlot(ba_loss, tr_loss, te_loss, ba_acc, tr_acc, te_acc, epochs,
               batch_losses, train_losses, test_losses, batch_accs, train_accs, test_accs):
	batch_x = np.linspace(max(min(epochs) + 1, 1), max(epochs) + 2, len(batch_losses)) - 0.5
	epochs = np.array(epochs) + 1
	ba_loss.data_source.data = {'x': batch_x, 'y': batch_losses}
	tr_loss.data_source.data = {'x': epochs, 'y': train_losses}
	te_loss.data_source.data = {'x': epochs, 'y': test_losses}
	ba_acc.data_source.data = {'x': batch_x, 'y': [1 - a for a in batch_accs]}
	tr_acc.data_source.data = {'x': epochs, 'y': [1 - a for a in train_accs]}
	te_acc.data_source.data = {'x': epochs, 'y': [1 - a for a in test_accs]}


def saveFigure(figFile, title, epochs, batch_losses, train_losses, test_losses, batch_accs, train_accs, test_accs):
	batch_x = np.linspace(max(min(epochs) + 1, 1), max(epochs) + 2, len(batch_losses)) - 0.5
	epochs = np.array(epochs) + 1
	batch_err = [1 - a for a in batch_accs]
	train_err = [1 - a for a in train_accs]
	test_err = [1 - a for a in test_accs]
	
	dpi = 80
	width = 1500 / dpi
	height = 500 / dpi
	fig = plt.figure(figsize=(width, height), dpi=dpi)
	plt.suptitle(title, fontsize=20)
	plt.subplot(1, 2, 1)
	plt.title('Training/Testing Loss per Epoch', fontsize=16)
	plt.xlabel('Epoch', fontsize=12)
	plt.ylabel('Loss', fontsize=12)
	plt.yscale('log')
	plt.plot(batch_x, batch_losses, color='skyblue', alpha=0.5, label='Training(Batch)', linewidth=0.1)
	plt.plot(epochs, train_losses, color='b', label='Training')
	plt.plot(epochs, test_losses, color='r', label='Testing')
	plt.legend()
	plt.grid(which='minor', linestyle=':')
	plt.grid(which='major', linestyle='-')
	
	plt.subplot(1, 2, 2)
	plt.title('Training/Testing Error per Epoch', fontsize=16)
	plt.xlabel('Epoch', fontsize=12)
	plt.ylabel('Error', fontsize=12)
	plt.yscale('log')
	plt.plot(batch_x, batch_err, color='skyblue', alpha=0.5, label='Training(Batch)', linewidth=0.1)
	plt.plot(epochs, train_err, color='b', label='Training')
	plt.plot(epochs, test_err, color='r', label='Testing')
	plt.legend()
	plt.grid(which='minor', linestyle=':')
	plt.grid(which='major', linestyle='-')
	
	plt.subplots_adjust(wspace=.15, hspace=0)
	
	fig.savefig(figFile + '.png', format='png', dpi=dpi, bbox_inches='tight')
	fig.savefig(figFile + '.eps', format='eps', dpi=dpi, bbox_inches='tight')
	
	plt.close('all')