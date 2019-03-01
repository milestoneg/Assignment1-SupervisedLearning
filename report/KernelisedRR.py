"""
Created on Mon·29·Oct·2018

@author: Yuan Gao (18064382)
"""
#-----------------------------------------------
import numpy as np


#method used to calculate kernel matrix K
def kernelMat(x_train, sigma):# arguments: array and scalar
	l = len(x_train)
	K = np.zeros((l,l))
	for i in range (0,l):
		K[i,i] = 1
		for j in range (i+1,l):
			K[i,j] = np.exp(-np.linalg.norm(x_train[i]-x_train[j])**2/(2*sigma**2))
			K[j,i] = K[i,j]
	K = np.mat(K)
	return K

#method used to train model and return an array of alpha
def fit(xi, sigma, gamma, y):
 	alpha = np.linalg.inv((kernelMat(xi,sigma) + gamma* len(xi) * np.identity(len(xi)))) @ y
 	#print(alpha)
 	return np.array(alpha).flatten()

#method used to calculate value Gaussian kernel 
def Gaussian_kernel(Xi,Xj,sigma): # arguments: arrays and scalar
	K = np.exp(-np.linalg.norm(Xi - Xj)**2/(2*sigma**2))
	return K

#method used to predict value
def estimate(x,xtest,sigma,alpha):
	y_e = np.zeros((len(xtest),1))
	for i in range(len(xtest)):
		for j in range(len(x)):
			y_e[i] += alpha[j]*Gaussian_kernel(x[j],xtest[i],sigma)
	return y_e

'''
#test code
if __name__ == '__main__':
	
	xi = np.array([[1,2,3],[3,2,1]])
	xj = np.array([[6,7,8],[3,2,2]])
	y = np.array([1,2])
	print(kernelMat(xi,xj,0.5))
	print(fit(xi,0.5,0.3,y))
	print(estimate(xi,xj,0.5,a))
'''