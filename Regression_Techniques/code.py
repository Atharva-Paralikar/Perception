import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import math

file_path = 'sample_dataset.csv'

def get_filedata(path):
	age = []
	cost = []
	with open(path) as file:
		reader = csv.reader(file)
		header = next(reader)
		for row in reader:
			if row == 0:
				row = 1
			age.append(int(row[0]))
			cost.append(float(row[6]))
		
		data = np.vstack((age,cost))
		# print(data)

	return data

######### Least Squares ###########

def cov(x,y):
	cov = (np.sum((x - np.mean(x))*(y-np.mean(y))))/(len(x)-1)
	return cov

def cov_matrix(x,y):
	C = np.array([[cov(x,x), cov(x,y)],[cov(y,x), cov(y,y)]])
	return C

def l_s(x,y):
	A = np.zeros((len(x),2))
	Y = np.zeros((len(y),1))
	for r in range(0,len(x)):
		A[r][0] = x[r]
		A[r][1] = 1
		Y[r][0] = y[r]
	alpha = np.dot((np.dot(np.linalg.inv(np.dot(A.T,A)),A.T)),Y)

	Y = alpha[0]*x + alpha[1]
	plt.plot(x,Y,color = "blue",label = "Ordinary Least squares" )
	

########## Total Least Squares #########

def moment(x,y):
	moment = np.sum((x-x.mean())*(y-y.mean()))
	return moment

def moment_matrix(x,y):
	M = np.array([[moment(x,x), moment(x,y)],[moment(y,x), moment(y,y)]])
	return M

def total_l_s(x,y):

	M_matrix = moment_matrix(x,y)
	EigenVal,EigenVec = np.linalg.eig(M_matrix)

	c = EigenVec[:,-1]
	d = c[0]*x.mean() + c[1]*y.mean()
	Y = (c[0]*x + d)/c[1]

	plt.plot(x,Y,color = "green",label = "Total least squares" )

########## RANSAC #################

def find_line(age,cost):

	x1,x2,y1,y2 = age[0], age[1], cost[0],cost[1]
	m = (y1 - y2) / (x1 - x2 + 1e-7)
	c = y1 - m * x1
	return m,c

def ransac(x,y):

	n = 17
	threshold = 1000
	max_inliers = 0

	for i in range(n):
		
		a = random.randint(0,(len(x)-1))
		b = random.randint(0,(len(x)-1))
		Points_age = np.array([x[a],x[b]])
		Points_cost = np.array([y[a],y[b]])
		m,c = find_line(Points_age,Points_cost)

		inliers = 0
		for xp,yp in list(zip(x,y)):
			distance = abs (yp - (m*xp + c))
			if (distance < 1000):
				inliers += 1
		if max_inliers < inliers:
			max_inliers = inliers
			ransac_model = [m,c]
	Y = ransac_model[0]*x + ransac_model[1]
	return Y	


if '__name__ == __main__':

	raw_data = (get_filedata(file_path))
	age = raw_data[0]
	cost = raw_data[1]

	l_s(age,cost)
	total_l_s(age,cost)
	Y = ransac(age,cost)

	C_matrix = cov_matrix(age,cost)
	print("Covariance matrix = ",C_matrix)
	EigenVal, EigenVec = np.linalg.eig(C_matrix)
	# print("EigenValues = ",EigenVal)
	# print("EigenVectors = ",EigenVec)
	evector1 = EigenVec[:,0]
	evector2 = EigenVec[:,1]

	origin = [age.mean(),cost.mean()]
	plt.plot(age,cost,'b.')
	
	plt.quiver(*origin, *evector1,color = ['r'],scale = 21)
	plt.quiver(*origin, *evector2,color = ['b'],scale = 21)
	plt.plot(age,Y,color = "red",label = "RANSAC" )
	plt.legend()
	plt.savefig('Comparison.png')
	plt.show()
	 
		
