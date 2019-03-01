import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from sklearn.model_selection import train_test_split,KFold
data  = (scio.loadmat('boston.mat'))['boston'] # 读取波士顿数据集
attribute_number = 13

X = (data)[:,range(attribute_number)] # 生成训练集和测试集
y = data[:,attribute_number]



# 初始化
number = 15
#c_range = np.arange(number)
c_range = 9
MSE = np.zeros(number)
MSE2 = np.zeros(number)

for k in range(number):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3 ,random_state =4)
    y_train = y_train.reshape(-1,1)

    l = X_train.shape[0]

    K = X_train @ X_train.T + c_range**k
    #print(K)
    #print(c_range[k])
    alpha = np.linalg.solve(K,y_train)     # 另一种算矩阵的方法
    #alpha = np.linalg.inv(K) @ y_train
    alpha = alpha.reshape(-1,1)
    estimator = X_train @ X_train.T @ alpha

    print(estimator)

    MSE[k] = np.sum(np.power(estimator - y_train,2))/ X_train.shape[0]

    w = np.linalg.solve(X_train.T @ X_train, X_train.T @ y_train)
    w = w.reshape(-1,1)

    estimator2 = np.dot(X_train, w)

    MSE2[k] = np.sum(np.power(estimator2 - y_train,2)) / X_train.shape[0]
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3 ,random_state =4)
y_train = y_train.reshape(-1,1)

l = X_train.shape[0]

K = X_train @ X_train.T + c_range[0]
#print(K)
#print(c_range[k])
alpha = np.linalg.solve(K,y_train)     # 另一种算矩阵的方法
#alpha = np.linalg.inv(K) @ y_train
alpha = alpha.reshape(-1,1)
estimator = X_train @ X_train.T @ alpha

print(estimator)

#MSE[k] = np.sum(np.power(estimator - y_train,2))/ X_train.shape[0]

w = np.linalg.solve(X_train.T @ X_train, X_train.T @ y_train)
w = w.reshape(-1,1)

estimator2 = np.dot(X_train, w)

#MSE2[k] = np.sum(np.power(estimator2 - y_train,2)) / X_train.shape[0]

'''
'''
plt.figure()                                                # 画100次之后，训练集的log平均误差
fig, ax = plt.subplots()
ax.plot(range(len(estimator)), estimator,label="predicted y")
ax.plot(range(len(estimator)), y_train,label="actual y in training set")
ax.legend()
plt.show()

'''



plt.figure()                                                # 画100次之后，训练集的log平均误差
fig, ax = plt.subplots()
ax.plot(range(k+1), MSE,label="run-100-log avg train error")
ax.legend()
plt.show()

plt.figure()                                                # 画100次之后，训练集的log平均误差
fig, ax = plt.subplots()
ax.plot(c_range, MSE2,label="run-100-log avg train error")
ax.legend()
plt.show()