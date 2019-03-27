#!/usr/env python3
# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import argparse
from matplotlib import pyplot as plt

import core

'''docstring'''


def reconstruct(pts1, pts2, cameraMatrix, distCoeffs, imageSize,
                R=None, T=None):
    '''实现从像素坐标到空间坐标的转换
    Parameters
    -----------

    pts1, pts2 : Nx1x2 array, N = 1,2,3...
    N为特征点的个数

    Reference
    -----------
    https://blog.csdn.net/Devecor/article/details/88380979
    https://docs.opencv.org/4.0.1/d9/d0c/group__calib3d.html
    '''
    assert cameraMatrix.shape == (2, 3, 3) and distCoeffs.shape == (2, 4), \
        '维度不对!'
    cameraMatrix1, cameraMatrix2 = cameraMatrix
    distCoeffs1, distCoeffs2 = distCoeffs
    del cameraMatrix, distCoeffs

    if R is None or T is None:
        K = 0.5 * (cameraMatrix1 + cameraMatrix2)
        E, mask_E = cv.findEssentialMat(pts1, pts2, K)
        retval, R, T, mask = cv.recoverPose(E, pts1, pts2, K)

    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv.stereoRectify(
        cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
        imageSize, R, T
    )

    # 将像素坐标转换为归一化坐标
    # pts1 = core.pixels2normalized(pts1, K)
    # pts2 = core.pixels2normalized(pts2, K)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='单/双目视觉的三维重建')
    parser.add_argument(
        '-K1', '--cameraMatrix1', metavar='float',
        type=float, nargs=4, help='相机内参1: fx, fy, cx, cy')
    parser.add_argument(
        '-K2', '--cameraMatrix2', metavar='float',
        type=float, nargs=4, help='相机内参2: fx, fy, cx, cy')
    parser.add_argument(
        '-kpair', '--kp_pair', metavar='keypoints',
        help='pass a npzfile, orb关键点文件')
    parser.add_argument(
        '-R', '--rotationMatrix', metavar='3x3 array',
        default=None, help='pass a npzfile, 相机的旋转矩阵')
    parser.add_argument(
        '-T', '--translationMatrix', metavar='1x3 array',
        default=None, help='pass a npzfile, 相机的平移矩阵')
    parser.add_argument('--distCoeffs1', metavar='float',
                        help='畸变参数: k1,k2,p1,p2')
    parser.add_argument('--distCoeffs2', metavar='float',
                        help='畸变参数: k1,k2,p1,p2')
    args = parser.parse_args('-kpair matches/s11-1-match-s11-1a.npy'.split())
# -K1 3123.8 3122.3 1497.6 2022.3
    # pts1, pts2 = args.pt1.pt, args.pt2.pt  matches/ref1-match-ref2.npy
    if args.cameraMatrix1 is None or args.cameraMatrix2 is None:
        K1 = core.getIntrinsicMat(core.K1)
        K2 = core.getIntrinsicMat(core.K2)
    else:
        K1 = core.getIntrinsicMat(args.cameraMatrix1)
        K2 = core.getIntrinsicMat(args.cameraMatrix2)
    K = 0.5 * (K1 + K2)
    if args.distCoeffs1 is None or args.distCoeffs2 is None:
        distCoeffs1 = core.distCoeffs1
        distCoeffs2 = core.distCoeffs2
    else:
        distCoeffs1 = np.array([float(i) for i in args.distCoeffs1.split(',')],
                               np.float64)
        distCoeffs2 = np.array([float(i) for i in args.distCoeffs2.split(',')],
                               np.float64)
    distCoeffs = np.array([distCoeffs1, distCoeffs2])

    filename = args.kp_pair
    pts1, pts2 = np.load(filename)
    assert len(pts1) > 0 and len(pts2) > 0, '2d点为空!'

    pts1, pts2, point3d, imageSize, P1, P2 = reconstruct(
        pts1, pts2, np.array([K1, K2]), distCoeffs, (3000, 4000))

    # 过滤3d点
    # filtered_p3d, filtered_pts1, filtered_pts2 = filter_p3d_dist(
    #     point3d, pts1, pts2, z_prob=570)
    filtered_p3d, filtered_pts1, filtered_pts2 = filter_p3d_dist(
        point3d, pts1, pts2)
    del pts1, pts2, point3d

    savefile = filename.split('.')[0] + '-rec3d'
    np.save(savefile, np.array([filtered_pts1, filtered_pts2, filtered_p3d]))

    p3d = np.array([i[0] for i in filtered_p3d], np.float64)
    p2d = np.array(filtered_pts1, np.float64)
    fig = plt.figure()
    ax1 = core.plot_scatter3d(p3d, fig, sub=(1, 2, 2))
    ax2 = core.plot_scatter2d(p2d, fig, sub=(1, 2, 1))
    core.plot_response_coord(ax1, ax2, fig)
    pass
