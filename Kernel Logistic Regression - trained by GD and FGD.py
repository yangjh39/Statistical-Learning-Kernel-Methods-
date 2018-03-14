# -*- coding: utf-8 -*-
# @Time    : 2018/3/2 11:14
# @Author  : Jiahao Yang
# @Email   : yangjh39@uw.edu

# ------------------------------------------------------------------------- #
#                               STAT 538 HW 8                               #
# ------------------------------------------------------------------------- #

# Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.feature_extraction import image
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import time
from pylab import *

# ------------------------------ Exercise 1 ------------------------------ #

def fetch_patches(x, p=5):

    """
    Fetch small patches in each image and scale data

    :param x: (3d array) the set of image matrix from which patches are fetched
    :param p: the parameter of patches' size (pixels)
    :param n: the number of patches we will fetch in each image
    :return: the matrix of patches
    """
    patches_matrix = []

    for i in np.arange(x.shape[0]):
        patches = []

        m = image.extract_patches_2d(x[i], (p, p))

        for j in np.arange(m.shape[0]):
            patch = m[j].reshape(1, p**2)[0]

            if ((patch.__pow__(2).sum())**0.5) != 0:
                patch /= (patch.__pow__(2).sum())**0.5

            patches.append(patch.tolist())

        patches_matrix.append(patches)

    return np.array(patches_matrix)

def images_kernel(x, y, sigma=1):
    """
    Compute the Gaussian(RBF) kernel between X and Y for each pair of rows x in X and y in Y.

    :param x: matrix1
    :param y: matrix2
    :param sigma: hyper-parameter in Gaussian kernel
    :return: value of kernel
    """

    # Considering the rows of X, Y as vectors, compute the distance matrix between each pair of vectors.(sklearn)
    xx = np.einsum('ij,ij->i', x, x)[:, np.newaxis]
    yy = np.einsum('ij,ij->i', y, y)[np.newaxis, :]
    k = x.dot(y.T)
    k *= -2
    k += xx
    k += yy
    k = np.maximum(k, 0.)

    # Gaussian Kernel
    k /= -(sigma**2.)
    k = np.exp(k)

    return k.sum()

def kernel_matrix(t, sigma=1):
    kv = np.array(np.meshgrid(np.arange(t.shape[0]), np.arange(t.shape[0])))[0]
    for i in np.arange(t.shape[0]):
        for j in np.arange(i, t.shape[0]):
            kv[i][j] = kv[j][i] = images_kernel(t[i], t[j], sigma)
        print(i)

    return kv

def kernel_matrix_train_test(train, test, sigma=1):
    kv = np.array(np.meshgrid(np.arange(test.shape[0]), np.arange(train.shape[0])))[0]
    for i in np.arange(train.shape[0]):
        for j in np.arange(test.shape[0]):
            kv[i][j] = images_kernel(train[i], test[j], sigma)
        print(i)

    return kv

def f(x, y, lbd, km):
    """
    This is the objective function of kernel logistic regression
    :param x: input vector
    :param y: sample labels vector
    :param lbd: penalty parameter
    :param km: input kernel matrix
    :return: function value
    """
    temp_vector = km.dot(x)
    n = len(y)
    one = np.repeat(1., n)

    func_value = one.dot(log(1.+exp(-y*temp_vector)))/n + lbd/2.*x.dot(temp_vector)

    return func_value

def f_gradient(x, y, lbd, km):
    """
    This is the gradient of objective function of kernel logistic regression
    :param x: input vector
    :param y: sample labels vector
    :param lbd: penalty parameter
    :param km: input kernel matrix
    :return: function value
    """

    temp_vector = km.dot(x)
    n = len(y)

    grad_value = np.diag(-y).dot(km.dot(exp(-y * temp_vector)/(1 + exp(-y * temp_vector))))/n + lbd * temp_vector

    return grad_value

def backtracking_line_search(x, y, lbd, km, bl_beta, t):
    """
    This is a function of using backtracking line search to
    find the optimal step size in gradient descent algorithm.
    :param x: The variable value in each GD iteration
    :param y: sample labels vector
    :param lbd: penalty parameter
    :param km: input kernel matrix
    :param bl_beta: shrinking parameter for backtracking line search
    :param t: initial step size
    :return: optimal step size
    """

    grad_value = f_gradient(x, y, lbd, km)
    grad_value_norm = grad_value.dot(grad_value)
    func_value = f(x, y, lbd, km)
    newx = x-t*grad_value

    while f(newx, y, lbd, km) >= func_value - 0.5*t*grad_value_norm:
        t = bl_beta*t
        newx = x-t*grad_value

    return t

def mysvm_grad(step, eps, y, lbd, km, bl_beta):
    """
    Perform Gradient descent Algorithm
    :param step: initial step size
    :param eps: stopping iteration accuracy
    :param y: sample labels vector
    :param lbd: penalty parameter
    :param km: input kernel matrix
    :param bl_beta: shrinking parameter for backtracking line search
    :return: optimal x that minimize objective function value
    """
    x = np.repeat(0., y.shape[0])
    grad_value = f_gradient(x, y, lbd, km)

    while grad_value.dot(grad_value) > eps:
        newstep = backtracking_line_search(x, y, lbd, km, bl_beta, step)
        grad_value = f_gradient(x, y, lbd, km)
        newx = x - newstep * grad_value
        x = newx

    return x

def mysvm_fast_grad(step, eps, y, lbd, km, bl_beta):
    """
    Perform Gradient descent Algorithm
    :param step: initial step size
    :param eps: stopping iteration accuracy
    :param y: sample labels vector
    :param lbd: penalty parameter
    :param km: input kernel matrix
    :param bl_beta: shrinking parameter for backtracking line search
    :return: optimal x that minimize objective function value
    """
    x = np.repeat(0., y.shape[0])
    beta = np.repeat(0., y.shape[0])
    t = 0
    grad_value = f_gradient(beta, y, lbd, km)

    a = np.random.normal(size=y.shape[0])
    b = np.random.normal(size=y.shape[0])
    grada = f_gradient(a, y, lbd, km)
    gradb = f_gradient(b, y, lbd, km)
    L = np.sqrt(np.power(grada - gradb, 2).sum()) / np.sqrt(np.power(x - y, 2).sum())
    tmin = min(step, bl_beta/L)
    newstep = step

    while grad_value.dot(grad_value) > eps:
        newstep = max(newstep*bl_beta, tmin)
        grad_value = f_gradient(beta, y, lbd, km)
        newx = beta - newstep * grad_value
        newbeta = newx + t/(t+3)*(newx - x)
        x = newx
        beta = newbeta
        t += 1
        print(t)
        print(f(x, y, lbd, km))
        if t > 10000:
            break

    return x

if __name__ == '__main__':
    # Import data
    lfw = datasets.fetch_lfw_people(min_faces_per_person=30)

    # Record the shape of plot
    h, w = lfw.images.shape[1:3]

    # label with 1(Jean Chretien)
    lfw_class1 = lfw.images[np.where(lfw.target == 15)]
    # label with -1(Luiz Inacio Lula da Silva)
    lfw_class2 = lfw.images[np.where(lfw.target == 23)]

    n_samples1 = lfw_class1.shape[0]
    n_samples2 = lfw_class2.shape[0]

    # Generate train and test set and fetch the patches from them
    train_class1 = fetch_patches(lfw_class1[0:np.floor(n_samples1 * .75).__int__()])
    train_class2 = fetch_patches(lfw_class2[0:np.floor(n_samples2 * .75).__int__()])
    test_class1 = fetch_patches(lfw_class1[np.floor(n_samples1 * .75).__int__():n_samples1])
    test_class2 = fetch_patches(lfw_class2[np.floor(n_samples2 * .75).__int__():n_samples2])

    train_data = np.append(train_class1, train_class2, 0)
    train_label = np.append(np.repeat(1, train_class1.shape[0]), np.repeat(-1, train_class2.shape[0]))
    test_data = np.append(test_class1, test_class2, 0)
    test_label = np.append(np.repeat(1, test_class1.shape[0]), np.repeat(-1, test_class2.shape[0]))

    # Train kernel logistic regression by fast gradient descent
    para_sigma = np.arange(0.05, 0.55, 0.05)
    para_lamda = np.arange(0.1, 1.1, 0.1)

    test_error = np.array(np.meshgrid(np.arange(para_sigma.shape[0]), np.arange(para_lamda.shape[0])))[0]*1.
    df = np.array(np.meshgrid(np.arange(para_sigma.shape[0]), np.arange(para_lamda.shape[0])))[0]*1.
    te_plot = []
    df_plot = []

    for i in np.arange(para_sigma.shape[0]):
        train_km = kernel_matrix(train_data, sigma=para_sigma[i])
        train_test_km = kernel_matrix_train_test(train_data, test_data, sigma=para_sigma[i])
        # pd.DataFrame(train_km).to_csv('trainkm'+str(i)+'.csv')
        # pd.DataFrame(train_test_km).to_csv('train_test_km' + str(i) + '.csv')

        # train_km = pd.read_csv('trainkm' + str(i)+'.csv')
        # train_km = np.array(train_km.iloc[:, 1::])
        # train_test_km = pd.read_csv('train_test_km' + str(i)+'.csv')
        # train_test_km = np.array(train_test_km.iloc[:, 1::])

        train_km = train_km/train_km.max()
        train_test_km = train_test_km/train_test_km.max()

        for j in np.arange(para_lamda.shape[0]):
            alpha_fgd = mysvm_fast_grad(1, 0.01, train_label, para_lamda[j], train_km, 0.5)
            prob = exp(train_test_km.T.dot(alpha_fgd))/(1 + exp(train_test_km.T.dot(alpha_fgd)))
            df[i][j] = np.trace(
                np.linalg.inv(train_km + para_lamda[j] * np.eye(train_km.shape[0], train_km.shape[0])).dot(train_km))
            if str(prob[0]) == 'nan':
                test_error[i][j] = None
            else:
                test_error[i][j] = sum(abs((prob > 0.5)*2-1 != test_label)) / prob.shape[0]
                te_plot.append(test_error[i][j])
                df_plot.append(df[i][j])


    # SVM of scikit-learn
    train_data_svm = train_data.reshape(train_data.shape[0], -1)
    test_data_svm = test_data.reshape(test_data.shape[0], -1)

    time_start = time.time()
    param_grid = {'C': [1e3, 1e2, 1e-4, 5e-1, 1],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(train_data_svm, train_label)
    time_end = time.time()
    print('Time spent: ', time_end - time_start, 's')
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)

    # SVM's Misclassification error
    svm_test_error = mean(clf.predict(test_data_svm) != test_label)

    # Plot of misclassification error versus the kernel hyper-parameters and the regularization parameter
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(np.repeat(para_sigma, 10).reshape(10, 10),
                    np.repeat(para_lamda, 10).reshape(10, 10).T, test_error, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel('hyper-parameter sigma in kernel')
    ax.set_ylabel('regularization parameter lambda')
    ax.set_zlabel('error rate')
    ax.set_title('Misclassfication error of kernel logistic regression')
    plt.show()
    plt.savefig('E:\\U Washington\\Study\\538\\HW\\hw3\\Python\\exercise1_1.png')

    # Misclassification error versus degree of freedom
    plt.figure(2)
    plt.plot(df[:, 0], test_error[:, 0])
    plt.xlabel('degree of freedom')
    plt.ylabel('misclassification error')
    plt.title('Miscalssification error versus degree of freedom')
    plt.show()
    plt.savefig('E:\\U Washington\\Study\\538\\HW\\hw3\\Python\\exercise1_2.png')





