#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''位置求解'''
import argparse
# from collections import deque

import numpy as np
import cv2 as cv

import core


def solve_pose(query, refer):
    pass


def filter_respond(query, refer):
    '''通过检索待查询图片与模型库的对应点来过滤'''
    pt1_q, pt2_q = query
    pt1_r, pt2_r, pt_3d = refer

    mask_q = [0 for i in range(len(pt1_q))]
    mask_r = [0 for i in range(len(pt1_r))]

    temp = 1
    for i, (ex, ey) in enumerate(pt1_q):
        if temp >= len(mask_r):  # 终止不必要的循环
            break
        for j, (jx, jy) in enumerate(pt1_r):
            if (ex, ey) == (jx, jy) and mask_q[i] == 0 and mask_r[j] == 0:
                mask_q[i] = temp
                mask_r[j] = temp
                temp = temp + 1
                break

    def inner_filter(obj, mask):
        temp = [obj[i] for i, e in enumerate(mask) if e != 0]
        return [temp[i-1] for i in mask if i != 0]

    pt1_q = np.reshape(inner_filter(pt1_q, mask_q), (-1, 2))
    pt2_q = np.reshape(inner_filter(pt2_q, mask_q), (-1, 2))
    pt1_r = np.reshape(inner_filter(pt1_r, mask_r), (-1, 2))
    pt2_r = np.reshape(inner_filter(pt2_r, mask_r), (-1, 2))
    pt_3d = np.reshape(inner_filter(pt_3d, mask_r), (-1, 3))
    std = len(pt_3d)
    assert len(pt1_q) == std and len(pt2_q) == std and len(pt1_r) == std and \
        len(pt2_r) == std, 'bug in function filter_respond'
    return pt1_q, pt2_q, pt1_r, pt2_r, pt_3d


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pose solving')
    parser.add_argument('query', help='待查询的图片模型文件')
    parser.add_argument('refer', help='参考图片模型文件')
    parser.add_argument('--cameraMatrix', metavar='fx, fy, cx, xy',
                        help='相机内参矩阵')
    parser.add_argument('--distCoeffs', help='相机畸变参数')
    args = parser.parse_args(
#         'matches/ref1-match-ref2.npy \
# matches/ref1-match-ref2-rec3d.npy'.split()
'matches/s11-1-match-s11-1a.npy \
matches/s11-1-match-s11-1a-rec3d.npy'.split()
    )
    'matches/s11-1-match-s11-1a.npy \
matches/s11-1-match-s11-1a-rec3d.npy'.split()
# 'matches/s11-1-match-t20-3.npy \
# matches/s11-1-match-s11-1a-rec3d.npy'.split()
    pt1_q, pt2_q = np.load(args.query)
    pt1_r, pt2_r, pt_3d = np.load(args.refer)

    pt1_q, pt2_q, pt1_r, pt2_r, pt_3d = filter_respond(
        (pt1_q, pt2_q), (pt1_r, pt2_r, pt_3d)
    )

    K = 0.5 * (core.getIntrinsicMat(core.K1) + core.getIntrinsicMat(core.K2))
    distCoeffs = 0.5 * (core.distCoeffs1 + core.distCoeffs2)

    retval, rvec, tvec, inliers = cv.solvePnPRansac(
        pt_3d, pt2_q, K, distCoeffs, flags=cv.SOLVEPNP_EPNP)

    rotation = cv.Rodrigues(rvec)  # 转换为旋转矩阵
    # 计算旋转矩阵特征值和特征向量
    # retval, eigenvalue, eigenvector = cv.eigen(rotation[0])
    # 计算平移向量的特征值
    # retval = cv.eigenNonSymmetric(tvec)
    world_pose = -np.dot(np.linalg.inv(rotation[0]), tvec)

    print('revc is\n{}'.format(rvec))
    print('tvec is\n{}'.format(tvec))
    print('rotation is\n{}'.format(rotation[0]))
    print('pose is\n{}'.format(world_pose))
    pass
