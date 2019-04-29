#!/usr/bin python3
# -*- coding: utf-8 -*-

'''优化三维重建'''

import numpy as np
import cv2 as cv


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


def fliter_inliers(pt1_q, pt2_q, pt1_r, pt2_r, pt_3d, inliers):
    pt1_q = np.array([pt1_q[i] for i in inliers], np.float64).reshape((-1,2))
    pt2_q = np.array([pt2_q[i] for i in inliers], np.float64).reshape((-1,2))
    pt1_r = np.array([pt1_r[i] for i in inliers], np.float64).reshape((-1,2))
    pt2_r = np.array([pt2_r[i] for i in inliers], np.float64).reshape((-1,2))
    pt_3d = np.array([pt_3d[i] for i in inliers], np.float64).reshape((-1,3))
    return pt1_q, pt2_q, pt1_r, pt2_r, pt_3d


def adj_iterate(pt1_q, pt2_q, pt1_r, pt2_r, pt_3d,
                K, distCoeffs, flags=cv.SOLVEPNP_EPNP, tag=3):
    if flags == cv.SOLVEPNP_ITERATIVE:
        rot = np.float64([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]])
        retval, rvec, tvec, inliers = cv.solvePnPRansac(
            pt_3d, pt1_q, K, distCoeffs, flags=flags,
            rvec=cv.Rodrigues(rot)[0], tvec=np.float64([0, 0, 0]),
            useExtrinsicGuess=True)
    else:
        retval, rvec, tvec, inliers = cv.solvePnPRansac(
            pt_3d, pt1_q, K, distCoeffs, flags=flags)

    pt1q, pt2q, pt1r, pt2r, pt3d = fliter_inliers(
        pt1_q, pt2_q, pt1_r, pt2_r, pt_3d, inliers)

    if pt1q.shape[0] <= 8 or tag <= 0:
        return retval, rvec, tvec, inliers
    else:
        tag -= 1
        return adj_iterate(pt1q, pt2q, pt1r, pt2r, pt3d,
                    K, distCoeffs, flags=flags, tag=tag)
