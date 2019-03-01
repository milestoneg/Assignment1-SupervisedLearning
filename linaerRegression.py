"""
Created on Fri·20·Oct·2018

@author: ucabyg5
"""
#------------------------------------------------------------------------------
import numpy as np

def pl_featured(x, k):
	x_mat = np.zeros(shape=(len(x),k))#matrix initialize
	#construct X matrix after applying feature map
	for element in range(len(x)):
		pl_list = []
		for i in range(k):
			pl_list.append(x[element]**i)
		x_mat[element] = pl_list
	return x_mat

def pl_featured_sin(x, k):
	x_mat = np.zeros(shape=(len(x),k))#matrix initialize
	#construct X matrix after applying feature map
	for element in range(len(x)):
		pl_list = []
		for i in range(1,k+1):
			pl_list.append(np.sin(i * np.pi * x[element]))
		x_mat[element] = pl_list
	return x_mat
'''
def fit(x, y, k):
	x_mat = pl_featured(x, k)#convert nparray to matrix
	y_mat = np.array(y).T#y transpose
	xTx = x_mat.T @ x_mat
	w = np.linalg.inv(xTx)@ x_mat.T @ y_mat#calculate w according to least square method
	return np.array(w.T).flatten()#convert matrix to list
'''
def fit(x,y,k):
	X = np.around(pl_featured(x, k).astype(np.float64), decimals = 6)
	#return (np.linalg.inv(X.T.dot(X)).dot(X.T)).dot(y)
	return np.linalg.inv(np.dot(X.T,X)).dot(X.T).dot(y)

def fit_sin(x, y, k):
	x_mat = np.mat(pl_featured_sin(x, k))#convert nparray to matrix
	y_mat = np.mat(y).T#y transpose
	xTx = x_mat.T @ x_mat
	w = xTx.I @ x_mat.T @ y_mat#calculate w according to least square method
	return np.array(w.T).flatten()#convert matrix to list

def estimate(x, w):
	return x @ w.T 
	#return w @ x.T 

def mean_squared_error(y, y_e):
	diff = list(map(lambda x: (x[0]-x[1])**2, zip(y, y_e)))
	mse = np.sum(diff)/len(diff)
	return mse

#test code
'''
if __name__ == '__main__':
	x1         = np.random.uniform ( 0 , 1 , 30)
	epsilon = np.random.normal (0 , 0.07**2 , 30)
	y1         = np.square((np.sin ( 2 * np.pi * x1 )))+epsilon
	p = pl_featured(x1,1)
	alpha = fit(x1,y1,18)
	print(p)
	print(alpha)
'''