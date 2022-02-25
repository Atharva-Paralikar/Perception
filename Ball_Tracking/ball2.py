import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

path_video = "ball_video2.mp4"
xdata =  np.array([])
ydata = np.array([])
coordinates = []

def generate_frames(path):

	if not os.path.exists("frames_ball2"):
		os.mkdir("frames_ball2")
	count = 1
	cap = cv2.VideoCapture(path)
	
	while (cap.isOpened()):
		success,img = cap.read()
		if success:
			cv2.imwrite("./frames_ball2/"+str(count)+".jpeg",img)
		else:
			break
		if cv2.waitKey(10) == 27:
		  	break
		count +=1
	cap.release()
	cv2.destroyAllWindows()
	return count

def read_frames(frame_num):
	path = "./frames_ball2/"+str(frame_num)+".jpeg"
	frame = cv2.imread(path)
	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	thresh, bw_frame = cv2.threshold(gray_frame,127,255,cv2.THRESH_BINARY)
	final_frame = np.invert(bw_frame)
	
	#cv2.imshow("out",final_frame)
	#cv2.waitKey(0)

	ret, thresh = cv2.threshold(final_frame,127,255,0)
	contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	pixels = np.where(thresh == 255)
	# print(np.asarray(pixels))
	
	xcord = np.mean(np.asarray(pixels[1]))
	ycord_max = 1600 - np.max(np.asarray(pixels[0]))
	ycord_min = 1600 - np.min(np.asarray(pixels[0]))
	
	# print(int (xcord),int (ycord))
	# cv2.imshow("out",thresh)
	# cv2.waitKey(100)
	
	centre = [int(xcord),int(ycord_min),int(ycord_max)]
	return centre

def least_square(x,y):
	A = np.zeros((len(x),3))
	Y = np.zeros((len(y),1))
	for r in range(0,len(x)):
		A[r][0] = x[r]**2
		A[r][1] = x[r]
		A[r][2] = 1
		Y[r][0] = y[r]
	
	alpha = np.dot((np.dot(np.linalg.inv(np.dot(A.T,A)),A.T)),Y)
	
	#print ("alpha = ",alpha)
	
	return alpha

def function(x,alpha):
	Y = A[0]*x**2 + A[1]*x + A[2]
	return Y

if '__name__ == __main__':
	c = generate_frames(path_video)

	for i in range (1,c):
		coordinates = read_frames(i)
		point1 = [coordinates[0],coordinates[1]]
		point2 = [coordinates[0],coordinates[2]]
		xdata = np.append(xdata, coordinates[0])
		xdata = np.append(xdata, coordinates[0])
		ydata = np.append(ydata, coordinates[1])
		ydata = np.append(ydata, coordinates[2])
	
	# print('xdata = ',xdata)
	# print('ydata = ',ydata)

	A = least_square(xdata,ydata)
	y = function(xdata,A)
	plt.plot(xdata,ydata,'b.', label = " Top and bottom most point of the Ball ")
	plt.plot(xdata,y,'r', label = " Least square approximation of the ball path")
	plt.savefig('ball2.png')
	plt.legend()
	plt.show()
