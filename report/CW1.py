"""
Created on Fri·20·Oct·2018

@author: Yuan Gao (18064382), Yuan Li (18057497)
"""
#-----------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import linaerRegression as lr
import scipy.io as scio
import KernelisedRR as krr
from sklearn.model_selection import train_test_split, KFold
from numpy.linalg import inv
from mpl_toolkits.mplot3d import Axes3D
from prettytable import PrettyTable

x = [1, 2, 3, 4]
y = [3, 2, 0, 5]

def part1_1_1_a():
    #define all variables as global variables
    global model_d1, model_d2, model_d3, model_d4
    
    #calculate w for dimension 1 to 4
    model_d1 = lr.fit(x, y, 1)
    model_d2 = lr.fit(x, y, 2)
    model_d3 = lr.fit(x, y, 3)
    model_d4 = lr.fit(x, y, 4)

    #configuration of plot
    plt.xlim(0, 5)
    plt.ylim(-4, 8)
    plt.xlabel('x')
    plt.ylabel('y')

    #draw plot
    x_axis1 = np.linspace(0, 5, num = 1000)
    y_axis1 = np.linspace(2.5, 2.5, num = 1000)
    '''
    y_axis2 = 1.5 + 0.4 * x_axis1
    y_axis3 = 9 - 7.1 * x_axis1 + 1.5 * x_axis1 ** 2
    y_axis4 = -5 + 15.17 * x_axis1 - 8.5 * x_axis1 ** 2 + 1.33 * x_axis1 ** 3 
    '''
    y_axis2 = lr.estimate(lr.pl_featured(x_axis1,2), model_d2)
    y_axis3 = lr.estimate(lr.pl_featured(x_axis1,3), model_d3)
    y_axis4 = lr.estimate(lr.pl_featured(x_axis1,4), model_d4)

    plt.scatter([1,2,3,4],[3,2,0,5])
    l1, = plt.plot(x_axis1, y_axis1, color='green', label='dimension 1')
    l2, = plt.plot(x_axis1, y_axis2, color='blue',  label='dimension 2')
    l3, = plt.plot(x_axis1, y_axis3, color='black', label='dimension 3')
    l4, = plt.plot(x_axis1, y_axis4, color='red',   label='dimension 4')
    # Place a legend to the right of this smaller subplot.
    plt.legend(handles = [l1,l2,l3,l4],labels = ['k=1','k=2','k=3','k=4'], loc='best')
    plt.show()

def part1_1_1_b():

    print('y1 = 2.5')
    print('y2 = 1.5 + 0.4 * x')
    print('y3 = 9 - 7.1 * x + 1.5 * x ** 2')
    print('y4 = -5 + 15.17 * x - 8.5 * x ** 2 + 1.33 * x ** 3') 

#This function is depend on part1_1_1_a
#part1_1_1_a should be run before this function
def part1_1_1_c():
    #calculate estimate value for different dimension
    estimate_y1 = lr.estimate(lr.pl_featured(x,1), model_d1)
    estimate_y2 = lr.estimate(lr.pl_featured(x,2), model_d2)
    estimate_y3 = lr.estimate(lr.pl_featured(x,3), model_d3)
    estimate_y4 = lr.estimate(lr.pl_featured(x,4), model_d4)

    #calculate MSEs
    mse1 = lr.mean_squared_error(y, estimate_y1)
    mse2 = lr.mean_squared_error(y, estimate_y2)
    mse3 = lr.mean_squared_error(y, estimate_y3)
    mse4 = lr.mean_squared_error(y, estimate_y4)

    print("MSE 1D: %f, MSE 2D: %f, MSE 3D: %f, MSE 4D: %f" % (mse1, mse2, mse3, mse4))
    
def part1_1_2_a_i():
    x1      = np.random.uniform ( 0 , 1    , 30 )
    epsilon = np.random.normal  ( 0 , 0.07 , 30 )
    g       = ( np.sin( 2 * np.pi * x1 )** 2 ) + epsilon 
    xrange  = np.linspace( 0 , 1 , num = 101 )
    
    #configuration of plot
    plt.xlim(0 , 1  )
    plt.ylim(0 , 1.1)
    plt.xlabel('x')
    plt.ylabel('y')
    #plot the data set and function
    plt.plot(xrange, ( np.sin ( 2 * np.pi * xrange ) ** 2 ), color='black' )
    plt.scatter( x1 , g )
    plt.show()
    
def part1_1_2_a_ii():
    x1        = np.random.uniform ( 0 , 1    , 30 )
    epsilon   = np.random.normal  ( 0 , 0.07 , 30 )
    y1        = ( np.sin( 2 * np.pi * x1 )** 2 ) + epsilon 
    model_d2  = lr.fit(x1, y1, 2)
    model_d5  = lr.fit(x1, y1, 5)
    model_d10 = lr.fit(x1, y1, 10)
    model_d14 = lr.fit(x1, y1, 14)
    model_d18 = lr.fit(x1, y1, 18)
    x_axis1   = np.linspace(0, 1, num=1000)
    # configuration of plot
    plt.xlim(0, 1)
    plt.ylim(-0.5, 1.5)
    plt.xlabel('x')
    plt.ylabel('y')


    
    estimate_y2 = lr.estimate(lr.pl_featured(x_axis1, 2), model_d2)
    l1, = plt.plot(x_axis1, estimate_y2, color='green', label='dimension 2')
    plt.scatter(x1, y1)
    # Place a legend to the right of this smaller subplot.
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #plt.show()

    
    estimate_y5 = lr.estimate(lr.pl_featured(x_axis1, 5), model_d5)
    l2, = plt.plot(x_axis1, estimate_y5, color='blue', label='dimension 5')
    #plt.scatter(x1, y1)
    # Place a legend to the right of this smaller subplot.
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #plt.show()

    
    estimate_y10 = lr.estimate(lr.pl_featured(x_axis1, 10), model_d10)
    l3, = plt.plot(x_axis1, estimate_y10, color='black', label='dimension 10')
    #plt.scatter(x1, y1)
    # Place a legend to the right of this smaller subplot.
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #plt.show()

    
    estimate_y14 = lr.estimate(lr.pl_featured(x_axis1, 14), model_d14)
    l4, = plt.plot(x_axis1, estimate_y14, color='red',   label='dimension 14')
    #plt.scatter(x1, y1)
    # Place a legend to the right of this smaller subplot.
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #plt.show()

    
    estimate_y18 = lr.estimate(lr.pl_featured(x_axis1, 18), model_d18)
    l5, = plt.plot(x_axis1, estimate_y18, color='purple', label='dimension 18')
    #plt.scatter(x1, y1)
    # Place a legend to the right of this smaller subplot.

    plt.legend(handles = [l1,l2,l3,l4,l5],labels = ['k=2','k=5','k=10','k=14','k=18'], bbox_to_anchor=(1.13,1.025),loc='upper right')
    plt.show()

#calculate MSE from 1d to 18d with original basis
def mse_1d_to_18d(x1, y1, xt, yt):
    mse = np.zeros(shape=(18,1))
    for k in range(1,19):
        #calculate w for dimension 1 to 18
        model_dk     = lr.fit(x1, y1, k)
        #calculate estimate value for different dimension
        estimate_yt  = lr.estimate(lr.pl_featured(xt,k), model_dk)
        #calculate MSEs
        mse[k-1]     = lr.mean_squared_error(yt, estimate_yt)
    return mse

#calculate MSE from 1d to 18d with new basis
def mse_1d_to_18d_sin(x1, y1, xt, yt):
    mse = np.zeros(shape=(18,1))
    for k in range(1,19):
        #calculate w for dimension 1 to 18
        model_dk     = lr.fit_sin(x1, y1, k)
        #calculate estimate value for different dimension
        estimate_yt  = lr.estimate(lr.pl_featured_sin(xt,k), model_dk)
        #calculate MSEs
        mse[k-1]         = lr.mean_squared_error(yt, estimate_yt)
    return mse

def part1_1_2_b():
    x1         = np.random.uniform ( 0 , 1    , 30 )
    epsilon = np.random.normal (0 , 0.07 , 30 )
    y1         = (np.sin ( 2 * np.pi * x1 ) ** 2)+epsilon
    #calculate mse between training set and fitting result from 1d to 18d
    mse = np.log(mse_1d_to_18d(x1,y1,x1,y1))
    x_axis1 = np.linspace(1, 18, num=18)
    
    # configuration of plot
    plt.xlim(1, 18)
    plt.xlabel('Dimension')
    plt.ylabel('log of training error')
    plt.plot(x_axis1, mse, color='purple', label='log of MSE vs k')
    
    # Place a legend to the right of this smaller subplot.
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

def part1_1_2_c():
    #training set
    x1         = np.random.uniform ( 0 , 1    , 30 )
    train_epsilon = np.random.normal (0 , 0.07 , 30 )
    y1         = (np.sin ( 2 * np.pi * x1 ) ** 2)+ train_epsilon
    #test set
    xc = np.random.uniform(0 , 1    , 1000 )
    test_epsilon = np.random.normal (0 , 0.07 , 1000 )
    gc         = ( np.sin( 2 * np.pi * xc )** 2 ) + test_epsilon 
    #calculate mse between test set and fitting result from 1d to 18d
    mse = np.log(mse_1d_to_18d(x1,y1,xc,gc))
    x_axis1 = np.linspace(1, 18, num=18)
    # configuration of plot
    plt.xlim(1, 18)
    plt.xlabel('Dimension')
    plt.ylabel('log of test error')
    plt.plot(x_axis1, mse, color='purple', label='log of MSE vs k with epsilon')
    # Place a legend to the right of this smaller subplot.
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

def part1_1_2_d():
    #loop b for 100 times
    mse = np.zeros(shape=(100,18))
    for i in range(100):
        x1 = np.random.uniform ( 0 , 1    , 30 )
        y1 = np.sin ( 2 * np.pi * x1 ) ** 2
        mse_inloop = mse_1d_to_18d(x1,y1,x1,y1)
        mse[i] = mse_inloop.T
    #calculate average for each column within mse matrix
    mse_meaned = np.log(np.mean(mse, axis=0))
    #plot 
    x_axis1 = np.linspace(1, 18, num=18)
    # configuration of plot
    plt.xlim(1, 18)
    plt.xlabel('Dimension')
    plt.ylabel('Log of 100 times average of training error')
    plt.plot(x_axis1, mse_meaned, color='purple', label='100 times average MSE from 1d to 18d')
    # Place a legend to the right of this smaller subplot.
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

    #loop c for 100 times
    mse_c = np.zeros(shape=(100,18))
    for i in range(100):
        #training set
        x1         = np.random.uniform ( 0 , 1    , 30 )
        train_epsilon = np.random.normal (0 , 0.07 , 30 )
        y1         = (np.sin ( 2 * np.pi * x1 ) ** 2)+ train_epsilon
        #test set
        xc = np.random.uniform(0 , 1    , 1000 )
        test_epsilon = np.random.normal (0 , 0.07 , 1000 )
        gc         = ( np.sin( 2 * np.pi * xc )** 2 ) + test_epsilon 

        mse_inloop2 = mse_1d_to_18d(x1,y1,xc,gc)
        mse_c[i] = mse_inloop2.T
    #calculate average for each column within mse_c matrix
    mse_c_meaned = np.log(np.mean(mse_c, axis=0))
    #plot 
    x_axis1 = np.linspace(1, 18, num=18)
    # configuration of plot
    plt.xlim(1, 18)
    plt.xlabel('Dimension')
    plt.ylabel('Log of 100 times average of test error')
    plt.plot(x_axis1, mse_c_meaned, color='purple', label='100 times average MSE from 1d to 18d with epsilon')
    # Place a legend to the right of this smaller subplot.
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

def part1_1_3():
    #experiment 2b with new basis
    x1         = np.random.uniform ( 0 , 1    , 30 )
    y1         = np.sin ( 2 * np.pi * x1 ) ** 2
    mse_b = np.log(mse_1d_to_18d_sin(x1,y1,x1,y1))
    x_axis1 = np.linspace(1, 18, num=18)
    # configuration of plot
    plt.xlim(1, 18)
    plt.xlabel('Dimension')
    plt.ylabel('Log of training error with basis sin(kπx)')
    plt.plot(x_axis1, mse_b, color='purple', label='log of MSE vs k with basis sin(kπx)')
    # Place a legend to the right of this smaller subplot.
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

    #experiment 2c with new basis
    #training set
    x1         = np.random.uniform ( 0 , 1    , 30 )
    train_epsilon = np.random.normal (0 , 0.07 , 30 )
    y1         = (np.sin ( 2 * np.pi * x1 ) ** 2)+ train_epsilon
    #test set
    xc = np.random.uniform(0 , 1    , 1000 )
    test_epsilon = np.random.normal (0 , 0.07 , 1000 )
    gc         = ( np.sin( 2 * np.pi * xc )** 2 ) + test_epsilon 
    
    mse_c = np.log(mse_1d_to_18d_sin(x1,y1,xc,gc))
    x_axis1 = np.linspace(1, 18, num=18)
    # configuration of plot
    plt.xlim(1, 18)
    plt.xlabel('Dimension')
    plt.ylabel('Log of test error with basis sin(kπx)')
    plt.plot(x_axis1, mse_c, color='purple', label='log of MSE vs k with epsilon with basis sin(kπx))')
    # Place a legend to the right of this smaller subplot.
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    
    #experiment 2d with new basis
    #loop b for 100 times
    mse_d_b = np.zeros(shape=(100,18))
    for i in range(100):
        x1 = np.random.uniform ( 0 , 1    , 30 )
        y1 = np.sin ( 2 * np.pi * x1 ) ** 2
        mse_inloop = mse_1d_to_18d_sin(x1,y1,x1,y1)
        mse_d_b[i] = mse_inloop.T
    #calculate average for each column within mse matrix
    mse_d_b_meaned = np.log(np.mean(mse_d_b, axis=0))
    #plot 
    x_axis1 = np.linspace(1, 18, num=18)
    # configuration of plot
    plt.xlim(1, 18)
    plt.xlabel('Dimension')
    plt.ylabel('Log of 100 times average training error with basis sin(kπx)')
    plt.plot(x_axis1, mse_d_b_meaned, color='purple', label='100 times average MSE from 1d to 18d with basis sin(kπx)')
    # Place a legend to the right of this smaller subplot.
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

    #loop c for 100 times
    mse_d_c = np.zeros(shape=(100,18))
    for i in range(100):
        #training set
        x1         = np.random.uniform ( 0 , 1    , 30 )
        train_epsilon = np.random.normal (0 , 0.07 , 30 )
        y1         = (np.sin ( 2 * np.pi * x1 ) ** 2)+ train_epsilon
        #test set
        xc = np.random.uniform(0 , 1    , 1000 )
        epsilon = np.random.normal (0 , 0.07 , 1000 )
        gc         = ( np.sin( 2 * np.pi * xc )** 2 ) + epsilon 
        
        mse_inloop2 = mse_1d_to_18d_sin(x1,y1,xc,gc)
        mse_d_c[i] = mse_inloop2.T
    #calculate average for each column within mse_c matrix
    mse_d_c_meaned = np.log(np.mean(mse_d_c, axis=0))
    #plot 
    x_axis1 = np.linspace(1, 18, num=18)
    # configuration of plot
    plt.xlim(1, 18)
    plt.xlabel('Dimension')
    plt.ylabel('Log of 100 times average test error with basis sin(kπx)')
    plt.plot(x_axis1, mse_d_c_meaned, color='purple', label='100 times average MSE from 1d to 18d with epsilon with basis sin(kπx)')
    # Place a legend to the right of this smaller subplot.
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    
    
def part1_2a():
    #load the data
    data                  = (scio.loadmat('boston.mat'))['boston']
    #spile the train data and test data
    data_train ,data_test = train_test_split(data,test_size=0.33) 
    #initialize the train and test sets
    x_train               = np.asmatrix(np.ones(len(data_train))).T
    x_test                = np.asmatrix(np.ones(len(data_test))).T
    mse_price_train       = np.zeros(20)
    mse_price_test        = np.zeros(20)
    for j in range(20):
        #fit the data with constant function
        w_price_train     = inv((x_train.T)@(x_train))@(x_train.T)@data_train[:,13]
        w_price_test      = inv((x_test.T)@(x_test))@(x_test.T)@data_test[:,13]
        #calculate MSEs
        mse_price_train[j]= lr.mean_squared_error(x_train*w_price_train, data_train[:,13])
        mse_price_test [j]= lr.mean_squared_error(x_test*w_price_test  , data_train[:,13])
        
        data_train ,data_test = train_test_split(data,test_size=0.33)   
    train_std = np.std(mse_price_train, ddof = 1)
    test_std = np.std(mse_price_test, ddof = 1)
    mse_price_train       = np.sum(mse_price_train)/20
    mse_price_test        = np.sum(mse_price_test )/20
    print('average MSE for train set\t',mse_price_train,'\naverage MSE for test set\t',mse_price_test)
    return mse_price_train, mse_price_test, train_std, test_std

def part1_2c():
    #load the data
    data                  = (scio.loadmat('boston.mat'))['boston']
    #spile the train data and test data
    data_train ,data_test = train_test_split(data,test_size=0.33)
    #initialize the MSE
    mse_price_train       = np.zeros((20,13))
    mse_price_test        = np.zeros((20,13))
    mse_mean_price_train  = np.zeros((1,13))
    mse_mean_price_test  = np.zeros((1,13))
    for j in range(20):
        for i in range(13):
            #initialize the train and test sets
            x_train              = data_train[:,i]
            x_test               = data_test [:,i]
            #add a bias term
            x_train              = np.asmatrix(np.vstack((x_train,np.ones(len(data_train))))).T#339*2
            x_test               = np.asmatrix(np.vstack((x_test,np.ones(len(data_test))))).T
            mse_price_train[j,i] = lr.mean_squared_error((x_train@(inv((x_train.T)@(x_train))@(x_train.T)@data_train[:,13]).T), data_train[:,13])
            mse_price_test [j,i] = lr.mean_squared_error((x_test@(inv((x_test.T)@(x_test))@(x_test.T)@data_test[:,13]).T), data_test[:,13])
        
        #random spilt the train and test set after each linear regression
        data_train ,data_test = train_test_split(data,test_size=0.33) 
    #take the average for results over 20 runs
    train_std = np.std(mse_price_train,axis = 0, ddof = 1)
    test_std = np.std(mse_price_test,axis = 0, ddof = 1)
    mse_mean_price_train = np.sum(mse_price_train,axis = 0)/20
    mse_mean_price_test  = np.sum(mse_price_test,axis = 0 )/20
    print("For liner regression with single attribute\t ")
    for i in range(len(mse_mean_price_train)):
        print("Linear regression attribute",i+1,"\tMSE train\t",mse_mean_price_train[i],"\tMSE test\t",mse_mean_price_test[i],"\n")
    return mse_mean_price_train, mse_mean_price_test, train_std, test_std

def part1_2d():
    #load the data
    data                  = (scio.loadmat('boston.mat'))['boston']
    #spile the train data and test data
    data_train ,data_test = train_test_split(data,test_size=0.33)
    #initialize the vector w and b
    mse_price_train = np.zeros(20)
    mse_price_test  = np.zeros(20)
    for j in range(20):
        #initialize the train and test sets
        x_train            = data_train[:,range(13)]
        x_test             = data_test [:,range(13)]
        #add a bias term
        x_train            = np.asmatrix(np.c_[(x_train,np.ones(len(data_train)))])
        x_test             = np.asmatrix(np.c_[(x_test ,np.ones(len(data_test )))])
        mse_price_train[j] = lr.mean_squared_error((x_train@((inv((x_train.T)@(x_train))@(x_train.T)@data_train[:,13]).T)),data_train[:,13])
        mse_price_test [j] = lr.mean_squared_error((x_test@((inv((x_test.T)@(x_test))@(x_test.T)@data_test[:,13]).T)),data_test[:,13])
        #random spilt the train and test set after each linear regression
        data_train ,data_test = train_test_split(data,test_size=0.33) 
    #calculate standard deviations on training set and testing set
    train_std = np.std(mse_price_train, ddof = 1)
    test_std = np.std(mse_price_test, ddof = 1)
    #take the average MSE for results over 20 runs on the train and test set 
    mse_price_train = np.sum(mse_price_train)/20
    mse_price_test  = np.sum(mse_price_test )/20
    print("Linear regression with all attributes\t",'\t MSE train\t',mse_price_train,'\tMSE test\t',mse_price_test)
    return mse_price_train, mse_price_test, train_std, test_std

def part1_3a():
    #load data from .mat file
    data = (scio.loadmat('boston.mat'))['boston']
    '''full data
    x = data[:,range(13)]
    y = data[:,13]
    '''
    data_train ,data_test = train_test_split(data,test_size=0.33)
    x_train = data_train[:,range(13)]
    y_train = data_train[:,13]
    x_test = data_test[:,range(13)]
    y_test = data_test[:,13]
    #x = np.array([[1,2,3,4],[5,6,7,8],[2,3,4,5],[7,8,9,10],[5,7,8,9]])
    #y = [1,2,3,4,5]
    global record_list, cv_mse_mat
    #construct gamma vector and sigma vector
    gamma = []
    sigma = []
    for i in range(-40, -25):
        gamma.append(2**i)
    for i in np.arange(7,13.5,0.5):
        sigma.append(2**i)
        
    #for all permutations do cross validation
    cv_mse_mat = np.zeros((len(gamma),len(sigma)))
    record_list = []
    counter = 0
    min_mse = float('inf')
    min_gamma_idx = 0
    min_sigma_idx = 0
    for i in range(len(gamma)):
        for j in range(len(sigma)):
            mean_mse = k_fold_crossvalidation(5, x_train, y_train, sigma[j], gamma[i])
            cv_mse_mat[i,j] = mean_mse
            record_list.append((gamma[i],sigma[j],mean_mse))
            if(mean_mse<min_mse):
                min_mse = mean_mse
                min_gamma_idx = i
                min_sigma_idx = j
            #code used to show progress
            counter = counter +1
            print('loop ' + str(counter) +', '+str((len(gamma)*len(sigma)-counter))+' left. '+'MSE: '+str(mean_mse))
    #print(mean_mse_list)
    #print(record_list)
    print('min_gamma_idx: %d, min_sigma_idx: %d, min_mse: %f'%(min_gamma_idx, min_sigma_idx, min_mse))
    file=open('data.txt','w')
    file.write(str(record_list));
    file.close()

def part1_3b():
    #plot 3D surface view
    #configuration of plot
    fig = plt.figure()
    #plot configuration
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Sigma')
    ax.set_ylabel('Gamma')
    ax.set_zlabel('Cross-validation error')
    plt.xlim(2**7,2**13)
    plt.ylim(2**(-40),2**(-26))
    
    sigma = []
    gamma = []
    mse = []
    for t in record_list:
        sigma.append(t[1])
        gamma.append(t[0])
        mse.append(t[2])
    ax.plot_trisurf(sigma, gamma, mse, cmap='rainbow')
    
    plt.show()

    #plot mat view
    plt.matshow(cv_mse_mat)
    plt.xlabel("Sigma")
    plt.ylabel("Gamma")
    plt.show()

def k_fold_crossvalidation(k, data_x, data_y, sigma, gamma):
    #split data into k segments averagely
    kf=KFold(n_splits=k)
    a = kf.split(data_x)
    mse_list = []
    for train_data_index, test_data_index in a:
        x_train = data_x[train_data_index]
        x_test = data_x[test_data_index]
        y_train = data_y[train_data_index]
        y_test = data_y[test_data_index]
        alpha = krr.fit(x_train, sigma, gamma, y_train)
        y_e = krr.estimate(x_train, x_test, sigma, alpha)
        mse = lr.mean_squared_error(y_test,y_e)
        mse_list.append(mse)
    sum = 0
    for i in range(len(mse_list)):
        sum+= mse_list[i]
    mean_mse = sum/len(mse_list)
    return mean_mse
    #alpha = krr.fit(data, data, sigma, gamma, y)

def part1_3c():
    #best performance parameters
    sigma = 2**10
    gamma = 2**(-31)
    #load data from .mat file
    data = (scio.loadmat('boston.mat'))['boston']
    #spile the train data and test data
    data_train ,data_test = train_test_split(data,test_size=0.33)
    x_train = data_train[:,range(13)]
    y_train = data_train[:,13]
    x_test = data_test[:,range(13)]
    y_test = data_test[:,13]
    #train model
    alpha = krr.fit(x_train, sigma, gamma, y_train)
    #calculate mse
    y_e_train = krr.estimate(x_train, x_train, sigma, alpha)
    mse_train = lr.mean_squared_error(y_train, y_e_train)
    y_e_test = krr.estimate(x_train, x_test, sigma, alpha)
    mse_test = lr.mean_squared_error(y_test, y_e_test)
    #print('MSE on training set is: %f MSE on test set is: %f'%(mse_train, mse_test))
    return mse_train, mse_test

def part1_3d():
    table = PrettyTable(["Method","MSE train","MSE test"])
    table.padding_width = 1

    #repeat 1.2a 20 times with random split data set
    mse_price_train, mse_price_test, std_train, std_test = part1_2a()
    table.add_row(["Naive Regression",str(mse_price_train)+' ± '+str(std_train),str(mse_price_test)+' ± '+str(std_test)])
    
    #repeat 1.2c 20 times with random split data set
    mse_price_train, mse_price_test, std_train, std_test = part1_2c()
    for i in range(len(mse_price_train)):
        table.add_row(["Linear Regression (attribute"+str(i+1)+")", str(mse_price_train[i])+' ± '+str(std_train[i]), str(mse_price_test[i])+' ± '+str(std_test[i])])

    #repeat 1.2d 20 times with random split data set
    mse_price_train, mse_price_test, std_train, std_test = part1_2d()
    table.add_row(["Linear Regression (all attributes)", str(mse_price_train) +' ± '+str(std_train),str(mse_price_test)+' ± '+str(std_test)])
    #repeat 5c 20 times with random split data set
    result_list_train_13c = []
    result_list_test_13c = []
    for i in range(20):
        mse_price_train, mse_price_test = part1_3c()
        result_list_train_13c.append(mse_price_train)
        result_list_test_13c.append(mse_price_test)
    train_std = np.std(result_list_train_13c, ddof = 1)
    test_std = np.std(result_list_test_13c, ddof = 1)
    result_list_train_13c = sum(result_list_train_13c)/20
    result_list_test_13c = sum(result_list_test_13c)/20
    table.add_row(["Kernel Ridge Regression", str(result_list_train_13c) +' ± '+str(train_std), str(result_list_test_13c) +' ± '+str(test_std)])
    print(table)

if __name__ == '__main__':
    #part1_1_1_a()
    #part1_1_1_b()
    #part1_1_1_c()
    #part1_1_2_a_i()
    #part1_1_2_a_ii()
    #part1_1_2_b()
    #part1_1_2_c()
    #part1_1_2_d()
    #part1_1_3()
    #part1_2a()
    #part1_2c()
    #part1_2d()
    #part1_3a()
    #part1_3b()
    #part1_3c()
    part1_3d()