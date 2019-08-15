import cv2
import numpy as np 
import os
import tensorflow as tf

import logging as log
import matplotlib.pyplot as plt
import common
from tensorflow.examples.tutorials.mnist import input_data

def res(x, pic):
	crop_size = (x, x)
	img = cv2.imread(pic)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img_new = cv2.resize(img, crop_size, interpolation = cv2.INTER_CUBIC)
	return img_new

def get_names(addr, filetype):
	'''
	Get the filenames from current location.

	Input:
	--------------
		addr : The addrector of the folder.

	Output:
	--------------
		None

	'''
	# print('[tips] We catch following names:', end=' ')
	names = []
	if not filetype=='': 
		for filename in os.listdir(addr):
			if filetype in filename:
				names.append(filename.replace(filetype,''))
				# print(filename.replace(filetype,''), end=' ')
	else:
		for filename in os.listdir(addr):
			if '.' not in filename:
				names.append(filename)
				# print(filename, end=' ')
	# print('.')
	return names

'''
建立模型
'''


SIZE = 64
x_data = tf.placeholder(tf.float32, [None, SIZE, SIZE, 3])
y_data = tf.placeholder(tf.float32, [None, None])

keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)

def weightVariable(shape):
	''' build weight variable'''
	init = tf.random_normal(shape, stddev=0.01)
	#init = tf.truncated_normal(shape, stddev=0.01)
	return tf.Variable(init)

def biasVariable(shape):
	''' build bias variable'''
	init = tf.random_normal(shape)
	#init = tf.truncated_normal(shape, stddev=0.01)
	return tf.Variable(init)

def conv2d(x, W):
	''' conv2d by 1, 1, 1, 1'''
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxPool(x):
	''' max pooling'''
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def dropout(x, keep):
	''' drop out'''
	return tf.nn.dropout(x, keep)

def cnnLayer(classnum):
	''' create cnn layer'''
	# 第一层
	W1 = weightVariable([3, 3, 3, 32]) # 卷积核大小(3,3)， 输入通道(3)， 输出通道(32)
	b1 = biasVariable([32])
	conv1 = tf.nn.relu(conv2d(x_data, W1) + b1)
	pool1 = maxPool(conv1)
	# 减少过拟合，随机让某些权重不更新
	drop1 = dropout(pool1, keep_prob_5) # 32 * 32 * 32 多个输入channel 被filter内积掉了

	# 第二层
	W2 = weightVariable([3, 3, 32, 64])
	b2 = biasVariable([64])
	conv2 = tf.nn.relu(conv2d(drop1, W2) + b2)
	pool2 = maxPool(conv2)
	drop2 = dropout(pool2, keep_prob_5) # 64 * 16 * 16

	# 第三层
	W3 = weightVariable([3, 3, 64, 64])
	b3 = biasVariable([64])
	conv3 = tf.nn.relu(conv2d(drop2, W3) + b3)
	pool3 = maxPool(conv3)
	drop3 = dropout(pool3, keep_prob_5) # 64 * 8 * 8

	# 全连接层
	Wf = weightVariable([8*16*32, 512])
	bf = biasVariable([512])
	drop3_flat = tf.reshape(drop3, [-1, 8*16*32])
	dense = tf.nn.relu(tf.matmul(drop3_flat, Wf) + bf)
	dropf = dropout(dense, keep_prob_75)

	# 输出层
	Wout = weightVariable([512, classnum])
	bout = weightVariable([classnum])
	#out = tf.matmul(dropf, Wout) + bout
	out = tf.add(tf.matmul(dropf, Wout), bout)
	return out

def train(train_x, train_y, tfsavepath):
	''' train'''
	log.debug('train')
	# print('!!!!!!!!!!!!!!!!!',train_y.shape)
	# out = cnnLayer(train_y.shape[1])
	out = cnnLayer(10)
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y_data))
	train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
	accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(y_data, 1)), tf.float32))

	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		batch_size = 10
		num_batch = len(train_x) // 10
		for n in range(10):
			r = np.random.permutation(len(train_x))
			train_x = train_x[r]
			train_y = train_y[r]
			# train_x = train_x[r, :]
			# train_y = train_y[r, :]

			for i in range(num_batch):
				batch_x = train_x[i*batch_size : (i+1)*batch_size]
				batch_y = train_y[i*batch_size : (i+1)*batch_size]
				_, loss = sess.run([train_step, cross_entropy],\
									feed_dict={x_data:batch_x, y_data:batch_y,
											keep_prob_5:0.75, keep_prob_75:0.75})

				print(n*num_batch+i, loss)

		# 获取测试数据的准确率
		acc = accuracy.eval({x_data:train_x, y_data:train_y, keep_prob_5:1.0, keep_prob_75:1.0})
		print('after 10 times run: accuracy is ', acc)
		saver.save(sess, tfsavepath)

def validate(test_x, tfsavepath):
	''' validate '''
	output = cnnLayer(2)
	#predict = tf.equal(tf.argmax(output, 1), tf.argmax(y_data, 1))
	predict = output

	saver = tf.train.Saver()
	with tf.Session() as sess:
		#sess.run(tf.global_variables_initializer())
		saver.restore(sess, tfsavepath)
		res = sess.run([predict, tf.argmax(output, 1)],
						feed_dict = {x_data: test_x,
									keep_prob_5:1.0, keep_prob_75: 1.0})
		return res

def onehot(numlist):
    ''' get one hot return host matrix is len * max+1 demensions'''
    b = np.zeros([len(numlist), max(numlist)+1])
    b[np.arange(len(numlist)), numlist] = 1
    return b.tolist()

def getfilesinpath(filedir):
    ''' get all file from file directory'''
    for (path, dirnames, filenames) in os.walk(filedir):
        for filename in filenames:
            if filename.endswith('.jpg'):
                yield os.path.join(path, filename)
        for diritem in dirnames:
            getfilesinpath(os.path.join(path, diritem))

def readimage(pairpathlabel):
    '''read image to list'''
    imgs = []
    labels = []
    for filepath, label in pairpathlabel:
        for fileitem in getfilesinpath(filepath):
            img = cv2.imread(fileitem)
            imgs.append(img)
            labels.append(label)
    return np.array(imgs), np.array(labels)

def getfileandlabel(filedir):
    ''' get path and host paire and class index to name'''
    dictdir = dict([[name, os.path.join(filedir, name)] \
                    for name in os.listdir(filedir) if os.path.isdir(os.path.join(filedir, name))])
                    #for (path, dirnames, _) in os.walk(filedir) for dirname in dirnames])

    dirnamelist, dirpathlist = dictdir.keys(), dictdir.values()
    indexlist = list(range(len(dirnamelist)))

    return list(zip(dirpathlist, onehot(indexlist))), dict(zip(indexlist, dirnamelist))
    

'''
savedpath =r'./faceImagesGray'
isExists = os.path.exists(savedpath)
if not isExists:
	os.makedirs(savedpath)

names = get_names('../TASK2/faceImagesGray', '')

for name in names:
	pics = get_names('../TASK2/faceImagesGray/%s' %name , '.jpg')
	savedpath =r'./faceImagesGray/' + name
	isExists = os.path.exists(savedpath)
	if not isExists:
		os.makedirs(savedpath)
	for pic in pics:
		path = '../TASK2/faceImagesGray/%s/%s.jpg' %(name, pic)
		cv2.imwrite('./faceImagesGray/%s/%s.jpg' %(name, pic), res(64, path))
	print('[Tips] %s\'s photos are resize successfully!' %name)
'''

pathlabelpair, indextoname = getfileandlabel('./faceImagesGray')
train_x, train_y = readimage(pathlabelpair)
train_x = train_x.astype(np.float32) / 255.0
train(train_x, train_y, './face.ckpt')