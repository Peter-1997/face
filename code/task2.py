import cv2
import os

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



print('[Begin!] Loading...')

names = get_names('../TASK1/faceImages', '')

# for name in names:
# 	os.mkdir('./faceImagesGray' + './%s' %name)


for name in names:

	savedpath =r'./faceImagesGray/' + name
	isExists = os.path.exists(savedpath)
	if not isExists:
		os.makedirs(savedpath)

	pics = get_names('../TASK1/faceImages/%s' %name , '.jpg')
	# print(len(pics))

	i = 0

	for pic in pics:
		# load images
		image = cv2.imread('../TASK1/faceImages/%s/%s.jpg' %(name, pic))
		# print(name)

		# Turn into Grey
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# Load feature classifier
		face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

		# Look for the face in the photo
		faces = face_cascade.detectMultiScale(
			gray,					# 要检测的图像
			scaleFactor = 1.15,		# 图像尺寸每次缩小的比例
			minNeighbors = 4,		# 一个目标至少要被检测多少次才能标记为人脸
			minSize = (120,120)		# 目标的最小尺寸
			)

		# Plot the rectangle
		for (x, y, w, h) in faces:
			# cv2.rectangle(image,(x,y),(x+w, y+h), (0,255,0), 2)
			cropped = gray[y:y+h, x:x+w]
			cv2.imwrite('./faceImagesGray/%s/%d.jpg' %(name, i), cropped)
			if h >=120 :
				i = i + 1

	print('[Tips] %s\'s photos are loaded successfully!' %name)
# cv2.imshow('image', image)
# cv2.waitKey()

