# -*- coding: utf-8 -*-
# @Time    : 2018/2/10 12:30
# @Author  : Jiahao Yang
# @Email   : yangjh39@uw.edu

# ------------------------------------------------------------------------- #
#                              STAT 538 Midterm                             #
# ------------------------------------------------------------------------- #

# Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
from sklearn import datasets
from sklearn.feature_extraction import image

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
    k *= -2.
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

def calculate_fn(k, t):
    n = k.shape[0]
    alpha = np.append(np.repeat(0., t), np.repeat(1., n-t))
    beta = np.append(np.repeat(1., t), np.repeat(0., n - t))

    return t*(n-t)/n*(1/(n-t)**2.*alpha.dot(k.dot(alpha)) - 2/(t*(n-t))*alpha.dot(k.dot(beta)) + 1/t**2*beta.dot(k.dot(beta)))

if __name__ == '__main__':
    # Import data
    lfw = datasets.fetch_lfw_people(min_faces_per_person=30)

    # Record the shape of plot
    h, w = lfw.images.shape[1:3]

    # label with 0(Jean Chretien)
    lfw_class1 = lfw.images[np.where(lfw.target == 15)]
    # label with 1(Serena Williams)
    lfw_class2 = lfw.images[np.where(lfw.target == 28)]

    n_samples1 = lfw_class1.shape[0]
    n_samples2 = lfw_class2.shape[0]

    ts_images = np.append(lfw_class1, lfw_class2, 0)
    ts_index = np.array(np.arange(0, n_samples1+n_samples2))
    ts_patches = fetch_patches(ts_images)

    # a) -----------------------------------------------------------
    # Display the first 100 images
    fig = plt.figure(figsize=(10, 10))
    columns = 10
    rows = 10
    for i in range(1, columns * rows + 1):
        img = ts_images[i-1]
        ax = fig.add_subplot(rows, columns, i)
        plt.imshow(img)
        plt.axis('off')
        ax.set_title(str(i)+'th image', fontsize=5)
    plt.show()

    # c) ------------------------------------------------------------------------
    km = kernel_matrix(ts_patches, 1)

    plt.imshow(km, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.xlabel('index of image')
    plt.ylabel('index of image')
    # plt.title('Heat plot of kernel matrix')
    plt.show()

    # e) --------------------------------------------------------------------
    t = np.arange(2, km.shape[0])
    fn = []
    for i in np.arange(len(t)):
        fn.append(calculate_fn(km, t[i]))

    plt.plot(t, fn)
    plt.plot(t[np.argmax(fn)], max(fn), 'r*')
    plt.ylim((0, max(fn)+10000))
    plt.vlines(t[np.argmax(fn)], 0, max(fn))
    plt.xlabel('t')
    plt.ylabel('Fn')

    # f) -----------------------------------------------------------------------
    parameter = np.arange(0.1, 1.1, 0.1)

    fn_matrix, t_hat, t_star = [], [], n_samples1

    for i in np.arange(len(parameter)):
        km = kernel_matrix(ts_patches, parameter[i])
        t = np.arange(2, km.shape[0])
        fn = []
        for j in np.arange(len(t)):
            fn.append(calculate_fn(km, t[j]))

        t_hat.append(t[np.argmax(fn)])
        fn_matrix.append(fn)
        print(str(9-i)+' iteration left.')

    fn_matrix = pd.read_csv('fn_matrix.csv')

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(np.arange(0.1, 1.1, 0.1), np.arange(2, 107))

    ax.plot_surface(X, Y, np.array(fn_matrix.iloc[:, 1::]).T, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel('hyper-parameter sigmal of Gaussian kernel')
    ax.set_ylabel('t')
    ax.set_zlabel('value of Fn(t)')
    # ax.set_title('Value of Fn(t) using improved Gaussian kernel')
    plt.show()

    t_hat = []
    for i in np.arange(fn_matrix.shape[0]):
        t_hat.append(t[int(np.argmax(fn_matrix.iloc[i, 1::]))])

    plt.plot(parameter, [abs(x-t_star) for x in t_hat])
    plt.xlabel('value of hyper parameter sigma')
    plt.ylabel('absolute value of t_hat - t_star')
























