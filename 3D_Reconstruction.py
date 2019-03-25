#!/usr/env python3
# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import argparse
from matplotlib import pyplot as plt

import core

'''docstring'''


def uv2XYZ(pts1, pts2, cameraMatrix, R, T, imageSize, **kwargs):
    '''实现从像素坐标到空间坐标的转换
    Parameters
    -----------

    pts1, pts2 : Nx1x2 array, N = 1,2,3...
    N为特征点的个数

    kwargs:
        distCoeffs : Nx1x4 array, N is 1 or 2

    Reference
    -----------
    https://blog.csdn.net/Devecor/article/details/88380979
    https://docs.opencv.org/4.0.1/d9/d0c/group__calib3d.html
    '''

    for kw in kwargs:
        if kw == 'distCoeffs':
            distCoeffs = np.reshape(kwargs[kw], (2, 1, 4))
            distCoeffs1, distCoeffs2 = distCoeffs
        else:
            break
    else:  # default values
        distCoeffs1, distCoeffs2 = np.float64([1] * 4), np.float64([1] * 4)

    if cameraMatrix.ndim == 2:
        cameraMatrix1 = cameraMatrix
        cameraMatrix2 = cameraMatrix
    elif cameraMatrix.ndim == 3:
        cameraMatrix1, cameraMatrix2 = cameraMatrix
    else:
        print('相机矩阵维度不对')

    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv.stereoRectify(
        cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
        imageSize, R, T
    )

    # # 将像素坐标转换为归一化坐标
    # pts1 = pixels2normalized(pts1, cameraMatrix)
    # pts2 = pixels2normalized(pts2, cameraMatrix)

    point4d = cv.triangulatePoints(P1, P2, pts1.T, pts2.T)
    point3d = cv.convertPointsFromHomogeneous(point4d.T)

    return pts1, pts2, point3d, imageSize, P1, P2


def filter_p3d_dist(p3d, pts1, pts2, z_prob=None, z_threshold_lim=1000,
                    z_threshold_prob=30):
    '''过滤3d空间点, 过滤条件为:
    1. 出现在照片中的空间点, 必在相机之前, 即 Z>0
    2. 室内场景的空间点在任意方向上不会超过10m
    3. 室内场景的特征点多集中于墙壁上, 即Z会在最可能的值附近

    parameters
    -----------
    p3d : array_like(Nx3)空间坐标序列
    pts1, pts2 : 对应的像素坐标序列
    z_prob : z的期望值
    z_threshold_lim : z的极限值
    z_threshold_prob : z_prob的边界
    '''
    def inner_if_zprob(z, z_prob, z_threshold_prob):
        if z > z_prob + z_threshold_prob or z < z_prob - z_threshold_prob:
            return False
        else:
            return True

    mask = [1 for i in range(len(p3d))]
    for i, e in enumerate(p3d):
        x, y, z = e[0]
        maxval = max(e[0])
        minval = min(e[0])
        if z < 0 or maxval > z_threshold_lim or minval < -z_threshold_lim:
            mask[i] = 0
        elif z_prob is not None:
            if not inner_if_zprob(z, z_prob, z_threshold_prob):
                mask[i] = 0

    filtered_p3d = [p3d[i] for i, e in enumerate(mask) if e == 1]
    filtered_pts1 = [pts1[i] for i, e in enumerate(mask) if e == 1]
    filtered_pts2 = [pts2[i] for i, e in enumerate(mask) if e == 1]
    return filtered_p3d, filtered_pts1, filtered_pts2


def pixels2normalized(points, cameraMatrix):
    '''将像素坐标转换为归一化坐标'''
    norm_coord = []
    for i, (x, y) in enumerate(points):
        nx = (x - cameraMatrix[0][2]) / cameraMatrix[0][0]
        ny = (y - cameraMatrix[1][2]) / cameraMatrix[1][1]
        norm_coord.append([nx, ny])
    return np.array(norm_coord, np.float32)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='单/双目视觉的三维重建')
    parser.add_argument(
        '-K1', '--cameraMatrix1', metavar='float',
        type=float, nargs=4, help='相机内参1: fx, fy, cx, cy')
    parser.add_argument(
        '-K2', '--cameraMatrix2', metavar='float',
        type=float, nargs=4, help='相机内参2: fx, fy, cx, cy')
    parser.add_argument(
        '-kp1', '--keypoints1', metavar='keypoints',
        help='pass a npzfile, orb关键点文件')
    parser.add_argument(
        '-kp2', '--keypoints2', help='pass a npzfile，orb关键点文件',
        metavar='keypoints')
    parser.add_argument(
        '-R', '--rotationMatrix', metavar='3x3 array',
        default=None, help='pass a npzfile, 相机的旋转矩阵')
    parser.add_argument(
        '-T', '--translationMatrix', metavar='1x3 array',
        default=None, help='pass a npzfile, 相机的平移矩阵')
    args = parser.parse_args()
# -K1 3123.8 3122.3 1497.6 2022.3
    # pts1, pts2 = args.pt1.pt, args.pt2.pt

    # fx, fy, cx, cy = args.cameraMatrix1

    f_x = 3048
    f_y = 3048
    c_x = 1500
    c_y = 2000

    K = np.array([[f_x, 0,  c_x],
                  [0,  f_y, c_y],
                  [0,  0,   1]], np.float64)

    R = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]], np.float64)

    T = np.array([[1], [0], [0]], np.float64)

    filename = 'matches/s11-1-match-s11-1a.npy'
    pts1, pts2 = np.load(filename)

    pts1, pts2, point3d, imageSize, P1, P2 = uv2XYZ(
        pts1, pts2, K.reshape((3, 3)), R, T, (3000, 4000))

    # 过滤3d点
    filtered_p3d, filtered_pts1, filtered_pts2 = filter_p3d_dist(
        point3d, pts1, pts2, z_prob=570)
    del pts1, pts2, point3d

    savefile = filename.split('.')[0] + '-rec3d'
    np.save(savefile, np.array([filtered_pts1, filtered_pts2, filtered_p3d]))

    p3d = np.array([i[0] for i in filtered_p3d], np.float32)
    p2d = np.array(filtered_pts1, np.float32)
    fig = plt.figure()
    ax1 = core.plot_scatter3d(p3d, fig, sub=(1, 2, 2))
    ax2 = core.plot_scatter2d(p2d, fig, sub=(1, 2, 1))
    core.plot_response_coord(ax1, ax2, fig)
    pass
