#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''位置求解'''

import argparse
# from collections import deque

import numpy as np
import cv2 as cv

import core
import adjustment as adj


def solve_pose(query, refer):
    pass


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

    pt1_q, pt2_q, pt1_r, pt2_r, pt_3d = adj.filter_respond(
        (pt1_q, pt2_q), (pt1_r, pt2_r, pt_3d)
    )

    K = 0.5 * (core.getIntrinsicMat(core.K1) + core.getIntrinsicMat(core.K2))
    distCoeffs = 0.5 * (core.distCoeffs1 + core.distCoeffs2)

    retval, rvec, tvec, inliers = adj.adj_iterate(
        pt1_q, pt2_q, pt1_r, pt2_r, pt_3d, K,
        distCoeffs, flags=cv.SOLVEPNP_ITERATIVE, tag=1)

    rotation = cv.Rodrigues(rvec)  # 转换为旋转矩阵
    nv = np.matrix(rotation[0]) * np.float64([0, 0, 1]).reshape(3, 1)
    yaw = np.arctan2(nv[0], nv[2])
    # 这是相对参考相机拍摄角度的偏转，顺时针为正值
    angle = yaw.A.ravel()[0] * 180 / np.pi
    pose = -tvec.ravel()

    theta_x, theta_y, theta_z = core.rotationMatrixToEulerAngles(rotation[0])
    yaw_ = theta_y * 180 / np.pi

    print('rvec is\n{}'.format(rvec))
    print('tvec is\n{}'.format(tvec))
    print('rotation is\n{}'.format(rotation[0]))
    print('')
    pass
