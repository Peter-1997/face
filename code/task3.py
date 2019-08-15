import os, cv2
import numpy as np

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

# 修改图片大小
def res(x, pic):
	crop_size = (x, x)
	img = cv2.imread(pic)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img_new = cv2.resize(img, crop_size, interpolation = cv2.INTER_CUBIC)
	return img_new

print('[Begin!] Loading...')
test_face = []
face = []
test_people = []
people = []

names = get_names('../TASK2/faceImagesGray', '')

for name in names:
	pics = get_names('../TASK2/faceImagesGray/%s' %name , '.jpg')
	for pic in pics:
		path = '../TASK2/faceImagesGray/%s/%s.jpg' %(name, pic)
		if int(pic) < 120 :
			test_face.append(res(64, path))
			test_people.append(name)
		else : 
			face.append(res(64, path))
			people.append(name)
	print('[Tips] %s\'s photos are resize successfully!' %name)

data = np.array([np.array(face), np.array(test_face)])
labels = np.array([np.array(people), np.array(test_people)])

np.save("data.npy", data)
np.save("labels.npy", labels)

print('[Finish!]')