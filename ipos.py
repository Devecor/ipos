#!/usr/bin python3
# -*- coding:utf-8 -*-

'''
Info
------
__author__: devecor
'''

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import getopt


def ratio_test(matches, ratio=0.7):
    # need to draw only good matches, so create a mask
    mask = [[0, 0] for i in range(len(matches))]

    # ratio test
    for i, (m, n) in enumerate(matches):
        if m.distance < ratio*n.distance:
            mask[i] = [1, 0]
    return mask


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ipos')
    parser.add_argument('--show', action='store_true', help='显示匹配结果')
    parser.add_argument('-o', '--output', default='matches', help='结果保存目录')
    parser.add_argument('--K', metavar='cx, cy, fx, fy', help='内参矩阵')
    args = parser.parse_args()

    # 读取图片

    ref_img1 = cv.imread('images/s11-1.jpg', flags=cv.IMREAD_GRAYSCALE)
    ref_img2 = cv.imread('images/s11-1a.jpg', flags=cv.IMREAD_GRAYSCALE)
    tes_img1 = cv.imread('images/t20-3.jpg', flags=cv.IMREAD_GRAYSCALE)
    tes_img2 = cv.imread('images/t20-4.jpg', flags=cv.IMREAD_GRAYSCALE)

    # 特征提取

    # 初始化
    orb = cv.ORB_create(nfeatures=8000)

    # 查询关键点并计算描述符
    ref_kp1, ref_des1 = orb.detectAndCompute(ref_img1, None)
    ref_kp2, ref_des2 = orb.detectAndCompute(ref_img2, None)
    tes_kp1, tes_des1 = orb.detectAndCompute(tes_img1, None)
    tes_kp2, tes_des2 = orb.detectAndCompute(tes_img2, None)

    # 根据BRIEF描述子进行匹配, 使用HAMMING距离

    # FLANN parameters
    # for orb
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,  # 12
                        key_size=12,  # 20
                        multi_probe_level=1)  # 2
    search_params = dict(checks=50)  # or pass empty dictionary

    # for SIFT or SURF
    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches1 = flann.knnMatch(tes_des1, ref_des1, k=2)
    matches2 = flann.knnMatch(tes_des2, ref_des2, k=2)

    # ratio test
    matchesMask1 = ratio_test(matches1)
    matchesMask2 = ratio_test(matches2)

    draw_params1 = dict(matchColor=(0, 255, 0),
                        singlePointColor=(255, 0, 0),
                        matchesMask=matchesMask1,
                        flags=cv.DrawMatchesFlags_DEFAULT)
    draw_params2 = dict(matchColor=(0, 255, 0),
                        singlePointColor=(255, 0, 0),
                        matchesMask=matchesMask2,
                        flags=cv.DrawMatchesFlags_DEFAULT)

    draw1 = cv.drawMatchesKnn(tes_img1, tes_kp1, ref_img1, ref_kp1, matches1,
                              None, **draw_params1)
    draw2 = cv.drawMatchesKnn(tes_img2, tes_kp2, ref_img2, ref_kp2, matches2,
                              None, **draw_params2)

    if args.show:
        # cv.imshow('draw1', draw1)
        # cv.imshow('draw2', draw2)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        plt.subplot(121)
        plt.imshow(draw1)
        plt.subplot(122)
        plt.imshow(draw2)
        plt.show()

    # 组装内参矩阵
    f_x = 3123.8
    f_y = 3122.3
    c_x = 1497.6
    c_y = 2022.3
    K = np.float64([[f_x,  0,  c_x],
                    [0,   f_y, c_y],
                    [0,    0,   1]])
    # 无过滤
    pt1 = np.float64([i.pt for i in tes_kp1])
    pt2 = np.float64([i.pt for i in ref_kp1])

    retval, mask = cv.findEssentialMat(pt1, pt2, K)

    E, K1, K2 = retval, K, K  # 待优化
    distCoeffs1, distCoeffs2 = np.float64([1] * 4), np.float64([1] * 4)

    retval, R, t, mask = cv.recoverPose(E, pt1, pt2, K)
    imageSize = (3000, 4000)

    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv.stereoRectify(
        K1, distCoeffs1, K2, distCoeffs2, imageSize, R, t)

    point4d = cv.triangulatePoints(P1, P2, pt1.T, pt2.T)

    point3d = cv.convertPointsFromHomogeneous(point4d.T)

    print('the R is\n {}'.format(R))
    print('the T is\n {}'.format(t))
    pass
