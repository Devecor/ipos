#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''some functions of tools'''
import os
import sys
import math
import logging

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import get_test_data
from matplotlib import cm
import cv2 as cv

f_x = 3048
f_y = 3048
c_x = 1500
c_y = 2000
K_zero = np.float64([[f_x,  0,  c_x],
                    [0,   f_y, c_y],
                    [0,    0,   1]])
distCoeffs_zero = np.array([0] * 4, np.float64)

K1 = '3178.2,3179.4,1497.5,2027.3'
K2 = '3130.4,3128.8,1487.6,2013.3'
distCoeffs1 = np.array([0.1692, -0.5744, 0, 0], np.float64) 
distCoeffs2 = np.array([0.1520, -0.4518, 0, 0], np.float64)


def get_filenames(path, name='*.txt'):
    '''
    搜索path下三层目录，返回文件名列表,name为正则表达式，默认值为'*.txt'
    return: list
    '''
    # name = '\'{}\''.format(name)
    if sys.platform == 'linux':
        return os.popen('find {dir} -maxdepth 3 -name {fileName}'.format(
            dir=path, fileName=name)).read().split('\n')[0:-1]
    elif sys.platform == 'win32':
        assert os.path.exists(path), '{dir} is not exit!'.format(dir=path)
        files = os.walk(path)
        name = os.path.splitext(name)[-1]
        aim_files = []
        for path, d, fileList in files:
            for fileName in fileList:
                if fileName.endswith(name):
                    aim_files.append(os.path.join(path, fileName))
        return aim_files


def mkdir_r(dir):
    path = dir.split('/')
    for i, e in enumerate(path):
        if isinstance(e, str):
            sep = '/' if sys.platform == 'linux' else '\\'
            os.system('mkdir {dir}'.format(dir=sep.join(path[0:i+1])))


def mk_dir(dir):
    '''按照给定目录创建'''
    path = dir.split('/')
    for i, e in enumerate(path):
        if isinstance(e, str):
            os.system('mkdir {dir}'.format(dir='/'.join(path[0:i+1])))


def plot_scatter3d(p3d, fig, show=False, sub=(1, 1, 1)):
    '''在三维坐标下绘制空间点
    parameter
    ----------
    fig : fig = plt.figure()
    '''
    assert len(p3d) > 0, 'make sure len(p3d) > 0 !'
    ax = fig.add_subplot(*sub, projection='3d')
    np.random.seed(19680801)

    xs, ys, zs = [], [], []
    for i in p3d:
        xs.append(int(i[0]))
        ys.append(int(i[1]))
        zs.append(int(i[2]))
    colors = np.random.rand(len(xs))
    max_x, min_x = max(xs), min(xs)
    max_y, min_y = max(ys), min(ys)
    max_z, min_z = max(zs), min(zs)
    print('points_max : {}\npoints_min : {}'.format((max_x, max_y, max_z),
                                                    (min_x, min_y, min_z)))
    ax.scatter(xs, ys, zs, c=colors, marker='.', s=10)
    ax.scatter(0, 0, 0, c='r', s=40)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if show:
        plt.show()
    return ax


def plot_scatter2d(p2d, fig, show=False, sub=(1, 1, 1)):
    '''绘制像素坐标散点图'''
    # Fixing random state for reproducibility
    np.random.seed(19680801)

    ax = fig.add_subplot(*sub)
    xs, ys = [], []
    for i in p2d:
        xs.append(int(i[0]))
        ys.append(int(i[1]) * -1)
    colors = np.random.rand(len(xs))

    ax.scatter(xs, ys, s=2, c=colors, alpha=0.5)
    ax.set_xlim(0, 3000)
    ax.set_ylim(-4000, 0)
    if show:
        plt.show()
    return ax


def plot_response_coord(ax1, ax2, fig):
    '''combination'''
    # fig.add_axes(ax1)
    # fig.add_axes(ax2)
    plt.show()


def getIntrinsicMat(string=None):
    f_x, f_y, c_x, c_y = [float(i) for i in string.split(',')]
    K = np.float64([[f_x,  0,  c_x],
                    [0,   f_y, c_y],
                    [0,    0,   1]])
    return K


def pixels2normalized(points, cameraMatrix):
    '''将像素坐标转换为归一化坐标'''
    norm_coord = []
    for i, (x, y) in enumerate(points):
        nx = (x - cameraMatrix[0][2]) / cameraMatrix[0][0]
        ny = (y - cameraMatrix[1][2]) / cameraMatrix[1][1]
        norm_coord.append([nx, ny])
    return np.array(norm_coord, np.float64)


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I_ = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I_ - shouldBeIdentity)
    return n < 1e-6


def rotationMatrixToEulerAngles(R):
    '''Calculates rotation matrix to euler angles
    The result is the same as MATLAB except the order
    of the euler angles ( x and z are swapped ).'''
    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def filter_perspective(pt1, pt2, K=None, kp1=None, kp2=None, matches=None, flags=0):
    '''根据匹配点在3维空间的透视关系过滤
    parameter
    ----------
    flags : 指定使用何种透视关系进行过滤
    '''
    Fundamental = 0
    Essential = 1
    Homography = 2
    if flags == 1 and K is None:
        temp = input('falgs为1时必须指定K! 是否使用flags=2,继续运行程序(Y/N)?\n')
        while True:
            if temp == 'Y' or temp == 'y':
                flags = 2
                break
            elif temp == 'N' or temp == 'n':
                sys.exit('程序退出！')
            else:
                sys.stdout.write('请输入Y/N:\n')
                temp = input()

    if flags == Fundamental:
        M, inlier = cv.findFundamentalMat(pt1, pt2, cv.FM_RANSAC, 2, 0.99)
    elif flags == Essential:
        M, inlier = cv.findEssentialMat(pt1, pt2, K, cv.RANSAC, 0.99, 2)
    elif flags == Homography:
        M, inlier = cv.findHomography(pt1, pt2, cv.RANSAC, 5)
    pt1 = np.float64([pt1[i] for i, e in enumerate(inlier) if e == 1])
    pt2 = np.float64([pt2[i] for i, e in enumerate(inlier) if e == 1])

    if kp1 is None or kp2 is None or matches is None:
        return pt1, pt2, None, None, None
    else:
        new_matches = []
        new_kp1, new_kp2 = [], []
        start_index = len(kp1)
        for i, e in enumerate(pt1):
            new_kp1.append(cv.KeyPoint(e[0], e[1], kp1[0].size))
            new_kp2.append(cv.KeyPoint(pt2[i][0], pt2[i][1], kp1[0].size))
            new_matches.append(cv.DMatch(i + start_index, i + start_index,
                                         0, 1))
        assert len(new_matches) == len(pt1), '遍历有bug'
        return pt1, pt2, kp1 + new_kp1, kp2 + new_kp2, matches + new_matches


def make_augmented_mat(m1, m2):
    '''构建增广矩阵'''
    return np.append(m1, m2, axis=1)


def isPrecisePt3d(pt, pt3d, K, R, t, max_err=3):
    '''判断2d点与3d点是否能够准确的对应, 返回掩码
    max_err : 最大允许误差
    '''
    pt3d = cv.convertPointsToHomogeneous(np.float64(pt3d))
    transition_mat = np.eye(3, 4)
    R_t_augm = np.append(np.append(R, t, axis=1),
                         np.float64([0, 0, 0, 1]).reshape((1, 4)), axis=0)
    anal_2d = []
    for i in pt3d:
        point_2d = K.dot(transition_mat.dot(R_t_augm.dot(i.T)))
        anal_2d.append(point_2d)
    anal_2d = cv.convertPointsFromHomogeneous(np.float64(anal_2d))
    mask = [0 for _ in range(len(pt))]
    for i, e in enumerate(pt):
        error = np.linalg.norm(anal_2d - e)
        if error < 3:
            mask[i] = 1
    logging.basicConfig(level=logging.INFO)
    logging.info('准确的3D点共有{}'.format(sum(mask)))
    return mask
