# -*- coding: utf-8 -*-
# @Time    : 2018/3/4 15:58
# @Author  : Jiahao Yang
# @Email   : yangjh39@uw.edu
# Packages
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from multiprocessing.dummy import Pool as ThreadPool
from matplotlib import pyplot as plt

# Kernel matrix based on gaussian RBF kernel
def gaussian_kernel_matrix(x, y, sigma=1):
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

    return k


# Power Iteration Algorithm
def power_iteration(A):
    n, d = A.shape

    v = np.ones(d) / np.sqrt(d)
    ev = v.dot(A.dot(v))

    while True:
        Av = A.dot(v)
        v_new = Av / np.linalg.norm(Av)

        ev_new = v_new.dot(A.dot(v_new))
        if np.abs(ev - ev_new) < 0.01:
            break

        v = v_new
        ev = ev_new

    return ev_new, v_new


# Simultaneous power iteration
def simultaneous_power_iteration(A, k):
    n, m = A.shape
    Q = np.random.rand(n, k)
    Q, _ = np.linalg.qr(Q)
    Q_prev = Q

    for i in range(1000):
        Z = A.dot(Q)
        Q, R = np.linalg.qr(Z)

        # can use other stopping criteria as well
        err = ((Q - Q_prev) ** 2).sum()

        Q_prev = Q
        if err < 1e-3:
            break

    return np.diag(R), Q


# Compute first k eigenvectors and eigenvalues
def k_principle_cal(X, k=1):
    """
    Calculate the first k principal components based on power iteration algorithm

    :param X: sample matrix
    :param k: the number of principal components
    :return: the first k principal components and their eigenvalues respectively
    """

    initial_component = pd.DataFrame(np.zeros(X.shape[1]))

    eigen_value = []
    eigen_vector = []

    for i in np.arange(k):
        Xnew = X - (X.dot(initial_component)).dot(initial_component.T)
        eva, eve = power_iteration(Xnew.T.dot(Xnew))

        eigen_value.append(eva)
        eigen_vector.append(eve.tolist())

        initial_component = pd.DataFrame(eve).copy()
        print(str(i+1)+'-th eigen value and eigen vector')

    return eigen_value, pd.DataFrame(eigen_vector).T


# Parallel calculation
def calculateParallel(layers, threads=2):
    pool = ThreadPool(threads)
    results = pool.map(cal_ed, layers)
    pool.close()
    pool.join()
    return results


# Calculate errorrate for each layer
def cal_ed(layer):
    # The value of kernel hyper-parameter sigma
    model_type = 'mlp'
    sigm = np.arange(5, 105, 5)

    # The number of principle components
    d = np.arange(1, 11, 1)

    # error rate
    ed = np.zeros([len(sigm), len(d)])

    images = np.array(pd.read_csv('f' + str(layer) + '_' + model_type + '.csv'))[:, 1::]
    labels = np.array(pd.read_csv('label_' + model_type + '.csv'))[:, 1::]
    for j in np.arange(len(sigm)):
        km = gaussian_kernel_matrix(images, images, sigma=sigm[j])
        km_centered = ((np.eye(km.shape[0]) - (1/km.shape[0])*np.ones([km.shape[0], km.shape[0]])).dot(km)).dot(
            np.eye(km.shape[0]) - (1/km.shape[0])*np.ones([km.shape[0], km.shape[0]]))
        #eigen_value, eigen_vector = k_principle_cal(km_centered, max(d))
        eigen_value, eigen_vector = simultaneous_power_iteration(km_centered, max(d))
        eigen_vector = pd.DataFrame(eigen_vector)
        for k in np.arange(len(d)):
            u_hat = np.array(eigen_vector.iloc[:, 0:d[k]])

            # Perform softmax to find beta
            logreg = LogisticRegression(C=10 ** 20, multi_class='multinomial', solver='sag')

            # Training Logistic regression
            logreg.fit(u_hat, np.argmax(labels, axis=1))

            ed[j, k] = np.mean(logreg.predict(u_hat) != np.argmax(labels, axis=1))

        print('Deep network:' + model_type + ', layer:' + str(layer) + ', sigma:' + str(sigm[j]))

    return ed


# Parallel calculation
def calculateParallel_cnn(layers, threads=2):
    pool = ThreadPool(threads)
    results = pool.map(cal_ed_cnn, layers)
    pool.close()
    pool.join()
    return results


# Calculate errorrate for each layer
def cal_ed_cnn(layer):
    model_type = 'cnn'

    # The value of kernel hyper-parameter sigma
    sigm = np.arange(5, 105, 5)

    # The number of principle components
    d = np.arange(1, 11, 1)

    # error rate
    ed = np.zeros([len(sigm), len(d)])

    images = np.array(pd.read_csv('f' + str(layer) + '_' + model_type + '.csv'))[:, 1::]
    labels = np.array(pd.read_csv('label_' + model_type + '.csv'))[:, 1::]
    for j in np.arange(len(sigm)):
        km = gaussian_kernel_matrix(images, images, sigma=sigm[j])
        km_centered = ((np.eye(km.shape[0]) - (1 / km.shape[0]) * np.ones([km.shape[0], km.shape[0]])).dot(km)).dot(
            np.eye(km.shape[0]) - (1 / km.shape[0]) * np.ones([km.shape[0], km.shape[0]]))
        # eigen_value, eigen_vector = k_principle_cal(km_centered, max(d))
        eigen_value, eigen_vector = simultaneous_power_iteration(km_centered, max(d))
        eigen_vector = pd.DataFrame(eigen_vector)
        for k in np.arange(len(d)):
            u_hat = np.array(eigen_vector.iloc[:, 0:d[k]])

            # Perform softmax to find beta
            logreg = LogisticRegression(C=10 ** 20, multi_class='multinomial', solver='sag')

            # Training Logistic regression
            logreg.fit(u_hat, np.argmax(labels, axis=1))

            ed[j, k] = np.mean(logreg.predict(u_hat) != np.argmax(labels, axis=1))

        print('Deep network:'+model_type+', layer:' + str(layer) + ', sigma:' + str(sigm[j]))

    return ed


if __name__ == "__main__":
    # The number of layers
    l = np.array([0, 1, 2, 3])

    ed_matrix_mlp = calculateParallel(l, 2)
    pd.DataFrame(ed_matrix_mlp[0]).to_csv('layer0_ed_mlp.csv')
    pd.DataFrame(ed_matrix_mlp[1]).to_csv('layer1_ed_mlp.csv')
    pd.DataFrame(ed_matrix_mlp[2]).to_csv('layer2_ed_mlp.csv')
    pd.DataFrame(ed_matrix_mlp[3]).to_csv('layer3_ed_mlp.csv')

    ed_matrix_cnn = calculateParallel_cnn(l, 2)
    pd.DataFrame(ed_matrix_cnn[0]).to_csv('layer0_ed_cnn.csv')
    pd.DataFrame(ed_matrix_cnn[1]).to_csv('layer1_ed_cnn.csv')
    pd.DataFrame(ed_matrix_cnn[2]).to_csv('layer2_ed_cnn.csv')
    pd.DataFrame(ed_matrix_cnn[3]).to_csv('layer3_ed_cnn.csv')

    ed_mlp0 = np.array(pd.read_csv('layer0_ed_mlp.csv').min(axis=0)[1::])
    ed_mlp1 = np.array(pd.read_csv('layer1_ed_mlp.csv').min(axis=0)[1::])
    ed_mlp2 = np.array(pd.read_csv('layer2_ed_mlp.csv').min(axis=0)[1::])
    ed_mlp3 = np.array(pd.read_csv('layer3_ed_mlp.csv').min(axis=0)[1::])

    ed_cnn0 = np.array(pd.read_csv('layer0_ed_cnn.csv').min(axis=0)[1::])
    ed_cnn1 = np.array(pd.read_csv('layer1_ed_cnn.csv').min(axis=0)[1::])
    ed_cnn2 = np.array(pd.read_csv('layer2_ed_cnn.csv').min(axis=0)[1::])
    ed_cnn3 = np.array(pd.read_csv('layer3_ed_cnn.csv').min(axis=0)[1::])

    d = np.arange(1, 11, 1)

    plt.figure(1)
    line0, = plt.plot(d, ed_mlp0, color='black', alpha=0.2)
    line1, = plt.plot(d, ed_mlp1, color='black', alpha=0.6)
    line2, = plt.plot(d, ed_mlp2, color='black', alpha=1)

    plt.xlabel('dimensionality d')
    plt.ylabel('error e(d)')
    # plt.ylim([0, 1])
    plt.legend([line0, line1, line2], ["Layer 0", "Layer 1", "Layer 2"])
    plt.title('Multilayer perceptron: MINIST-10K')

    plt.figure(2)
    line0, = plt.plot(d, ed_cnn0, color='black', alpha=0.2)
    line1, = plt.plot(d, ed_cnn1, color='black', alpha=0.6)
    line2, = plt.plot(d, ed_cnn2, color='black', alpha=1)

    plt.xlabel('dimensionality d')
    plt.ylabel('error e(d)')
    # plt.ylim([0, 1])
    plt.legend([line0, line1, line2], ["Layer 0", "Layer 1", "Layer 2"])
    plt.title('Convolutional neural network: MINIST-10K')

    d10_mlp = np.array([ed_mlp0[-1], ed_mlp1[-1], ed_mlp2[-1], ed_mlp3[-1]])
    d10_cnn = np.array([ed_cnn0[-1], ed_cnn1[-1], ed_cnn2[-1], ed_cnn3[-1]])

    plt.figure(3)
    line3, = plt.plot([0, 1, 2, 3], d10_mlp, 'bo-')
    line4, = plt.plot([0, 1, 2, 3], d10_cnn, 'r^-')

    plt.xlabel('layer l')
    plt.ylabel('error e(d)')
    # plt.ylim([0, 1])
    plt.legend([line3, line4], ["MLP", "CNN"])
    plt.title('Supervised learning: MINIST-10K')



