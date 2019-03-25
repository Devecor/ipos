#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''位置求解'''
import argparse

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

    for i, (ex, ey) in enumerate(pt1_q):
        for j, (jx, jy) in enumerate(pt1_r):
            if (ex, ey) == (jx, jy):
                mask_q[i] = 1
                mask_r[j] = 1

    def inner_filter(obj, mask):
        return [obj[i] for i, e in enumerate(mask) if e == 1]

    pt1_q = inner_filter(pt1_q, mask_q)
    pt2_q = inner_filter(pt2_q, mask_q)
    pt1_r = inner_filter(pt1_r, mask_r)
    pt2_r = inner_filter(pt2_r, mask_r)
    pt_3d = inner_filter(pt_3d, mask_r)
    return pt1_q, pt2_q, pt1_r, pt2_r, pt_3d


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pose solving')
    parser.add_argument('query', help='待查询的图片模型文件')
    parser.add_argument('refer', help='参考图片模型文件')
    parser.add_argument('--cameraMatrix', metavar='fx, fy, cx, xy',
                        help='相机内参矩阵')
    parser.add_argument('--distCoeffs', help='相机畸变参数')
    args = parser.parse_args(
        'matches/s11-1-match-t20-3.npy \
matches/s11-1-match-s11-1a-rec3d.npy'.split()
    )

    pt1_q, pt2_q = np.load(args.query)
    pt1_r, pt2_r, pt_3d = np.load(args.refer)

    pt1_q, pt2_q, pt1_r, pt2_r, pt_3d = filter_respond(
        (pt1_q, pt2_q), (pt1_r, pt2_r, pt_3d)
    )

    K = core.K_zero
    distCoeffs = core.distCoeffs_zero
    retval, rvec, tvec = cv.solvePnP(
        np.array(pt_3d, np.float32), np.array(pt2_q, np.float32), K,
        distCoeffs, flags=cv.SOLVEPNP_ITERATIVE)
    # print(retval)
    # print(rvec)
    # print(tvec)
    rotation = cv.Rodrigues(rvec)
    print(np.linalg.det(rotation[0]))
    pass
