#!/usr/bin python3
# -*- coding:utf-8 -*-

'''
使用orb匹配照片，使用tatio_test进行过滤，直接进行3维重建并可视化， 未进行
Info
------
__author__: devecor
'''

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import argparse
import glob

import core


def ratio_test(matches, ratio=0.75):
    '''距离比值过滤: 描述符距离相近的点对, 无法确定哪一个匹配正确
    paramters
    ----------
    matches : 输入匹配结果

    ratio : 过滤的条件, 默认为0.75

    Return
    ----------
    mask : array_like 返回一组表示是否正确匹配的掩码
    '''
    # need to draw only good matches, so create a mask
    mask = [[0, 0] for i in range(len(matches))]

    # ratio test
    for i, e in enumerate(matches):
        if len(e) <= 2:
            continue
        m, n = e[0:2]
        if m.distance < ratio*n.distance:
            mask[i] = [1, 0]

    matches = [matches[i] for i, e in enumerate(mask) if e == [1, 0]]
    return matches, mask


def detect(img, feature='orb', n=8000):
    '''提取局部特征, 默认为orb, 默认提取8000个特征点'''
    # 初始化
    orb = cv.ORB_create(nfeatures=n)  # 拟提取8000个特征

    # 查询关键点并计算描述符
    kp, des = orb.detectAndCompute(img, None)

    return kp, des, img


def show_matches(img1, kp1, img2, kp2, matches, mask=None):
    '''展示匹配结果
    parameters
    -----------
    img1, img2 : 图片矩阵

    kp1, kp2 : 对应的关键点

    matches : 匹配结果

    mask ： 匹配结果的掩码, 默认为None
    '''
    draw_params = dict(
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        matchesMask=mask,
        flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

    draw = cv.drawMatches(
        img1, kp1, img2, kp2, matches, None, **draw_params)

    # cv.imshow('draw1', draw1)
    # cv.imshow('draw2', draw2)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    plt.imshow(draw)
    plt.show()


def save_matches_to_img(img1, kp1, img2, kp2, matches, mask=None, config=None):
    '''将匹配结果保存到图片'''
    draw_params = dict(
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        matchesMask=mask,
        flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    draw = cv.drawMatches(
        img1, kp1, img2, kp2, matches, None, **draw_params)
    imgname = config.img[0].split('/')[-1].split('.')[0] + '-match-' + \
        config.img[1].split('/')[-1]
    path = config.output + '/' + imgname
    ifsaved = cv.imwrite(path, draw)
    print('保存匹配结果到图片{}'.format(path) if ifsaved else '{}目录不存在'.format(path))
    return path


def match(img1, img2, config, kpdes1=None, kpdes2=None,
          useProvide=False):
    '''对两张照片的特征点进行匹配
    Parameters
    -----------
    img1, img2 : array_like, 表示图片的mat对象
    config : 配置参数的自定义对象
    kpdes1, kpdes2 : (kp, des), 相应图片特征点和描述符
    useProvide : boolean, 是否使用提供的特征点和描述符(目前还未实现)

    Notice:
    --------
    config 目前只支持config.show=boolean
    '''

    kp1, des1, img1 = detect(img1, n=config.nfeatures)
    kp2, des2, img2 = detect(img2, n=config.nfeatures)

    # 根据BRIEF描述子进行匹配, 使用HAMMING距离

    # FLANN parameters
    # for orb
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,  # 12
                        key_size=12,  # 20
                        multi_probe_level=1)  # 2
    search_params = dict(checks=100)  # or pass empty dictionary

    # for SIFT or SURF
    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=200)

    return matches, kp1, kp2, des1, des2


def filter_match(kp1, kp2, matches, ratio=0.75):
    '''对匹配结果进行过滤, 将改变匹配结果
    parameters
    -----------
    kp1, kp2 : 关键点列表

    matches : 原始匹配结果

    ratio : 过滤条件, 默认为0.75

    return
    --------
    pt1, pt2 : 过滤后的关键点对应像素坐标序列

    kp_pairs : 正确匹配的关键点对列表

    mask : 表示是否正确匹配的掩码

    filtered_matches : 过滤之后的匹配结果
    '''
    mkp1, mkp2 = [], []
    mask = [[0, 0] for i in range(len(matches))]
    for i, m in enumerate(matches):
        if len(m) >= 2 and m[0].distance < m[1].distance * ratio:
            mkp1.append(kp1[m[0].queryIdx])
            mkp2.append(kp2[m[0].trainIdx])
            mask[i] = [1, 0]
    pt1 = np.float64([kp.pt for kp in mkp1])
    pt2 = np.float64([kp.pt for kp in mkp2])
    kp_pairs = list(zip(mkp1, mkp2))
    filtered_matches = [e[0] for i, e in enumerate(matches)
                        if mask[i] == [1, 0]]
    return pt1, pt2, list(kp_pairs), mask, filtered_matches


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='detect_match')
    parser.add_argument('img', nargs=2, metavar='image', help='一对照片')
    parser.add_argument('-nfeatures', default=8000, type=int, help='提取特征点数')
    parser.add_argument('--show', action='store_true', help='显示匹配结果')
    parser.add_argument('--save', action='store_true', help='是否保存结果')
    parser.add_argument('--output', metavar='dir', help='输出路径')
    parser.add_argument('--K1', metavar='float', help='内参矩阵：cx, cy, fx, fy')
    parser.add_argument('--K2', metavar='float', help='内参矩阵：cx, cy, fx, fy')
    parser.add_argument('--distCoeffs1', metavar='float', help='畸变参数: k1,k2,p1,p2')
    parser.add_argument('--distCoeffs2', metavar='float', help='畸变参数: k1,k2,p1,p2')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args(
        'images/A/s11-1.jpg images/A/s11-1a.jpg --save --output matches \
--K1 3178.2,3179.4,1497.5,2027.3 --K2 3130.4,3128.8,1487.6,2013.3 --debug \
--distCoeffs1 0.1692,-0.5744,0,0 \
--distCoeffs2 0.1520,-0.4518,0,0'.split()
        )
# --K 3123.8,3122.3,1497.6,2022.3
# ' --save --output matches \images/ref1.jpg images/ref2.jpg
# --K 3048,3048,1500,2000 --debug'.split()

    # 读取图片
    img1, img2 = args.img
    ref_img1 = cv.imread(img1, flags=cv.IMREAD_GRAYSCALE)
    ref_img2 = cv.imread(img2, flags=cv.IMREAD_GRAYSCALE)
    # tes_img1 = cv.imread('images/T/t20-3.jpg', flags=cv.IMREAD_GRAYSCALE)
    # tes_img2 = cv.imread('images/T/t20-4.jpg', flags=cv.IMREAD_GRAYSCALE)
    assert ref_img1 is not None and ref_img2 is not None, '图片不存在!'

    # 特征提取与匹配
    matches, kp1, kp2, des1, des2 = match(ref_img1, ref_img2, args)
    # matches_t, kp1_t, kp2_t, des1_t, des2_t = match(
    #     ref_img1, tes_img1, args
    # )

    # 过滤
    pt1, pt2, kp_pairs, mask, filtered_matches = filter_match(
        kp1, kp2, matches.copy(), ratio=0.5)
    # pt1_t, pt2_t, kp_pairs_t, mask_t, filtered_matches_t = filter_match(
    #     kp1_t, kp2_t, matches_t.copy(), ratio=0.6
    # )

    if args.show:
        show_matches(ref_img1, kp1, ref_img2, kp2, filtered_matches, mask=None)

    # 保存过滤结果到图片
    filtered_res_path = save_matches_to_img(ref_img1, kp1, ref_img2, kp2,
                                            filtered_matches, config=args)
    # filtered_res_path_t = save_matches_to_img(ref_img1, kp1_t, tes_img1, kp2_t,
    #                                           matches_t, config=args)
    # 保存过滤结果到文件
    np.save(filtered_res_path.split('.')[0], (pt1, pt2))
    # np.save('matches/s11-1-match-t20-3', (pt1_t, pt2_t))
    # 组装内参矩阵
    K1 = core.getIntrinsicMat(args.K1)
    K2 = core.getIntrinsicMat(args.K2)
    K = 0.5 * (K1 + K2)

    E, mask_E = cv.findEssentialMat(pt1, pt2, K)

    H, mask_H = cv.findHomography(pt1, pt2, cv.RANSAC, 5.0)
    dst = cv.perspectiveTransform(pt2.reshape(-1, 1, 2), H)

    distCoeffs1 = np.array([float(i) for i in args.distCoeffs1.split(',')], np.float64)
    distCoeffs2 = np.array([float(i) for i in args.distCoeffs2.split(',')], np.float64)

    retval, R, t, mask = cv.recoverPose(E, pt1, pt2, K)
    imageSize = (3000, 4000)
    ############################################
    # R = np.array([[1, 0, 0],
    #               [0, 1, 0],
    #               [0, 0, 1]], np.float64)
    # t = np.array([1, 0, 0], np.float64)
    ############################################

    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv.stereoRectify(
        K1, distCoeffs1, K2, distCoeffs2, imageSize, R, t)

    point4d = cv.triangulatePoints(P1, P2, pt1.T, pt2.T)

    point3d = cv.convertPointsFromHomogeneous(point4d.T)

    p3d = [i[0] for i in point3d]
    p2d = pt1
    fig = plt.figure()
    ax1 = core.plot_scatter3d(p3d, fig, sub=(1, 2, 2))
    ax2 = core.plot_scatter2d(p2d, fig, sub=(1, 2, 1))
    core.plot_response_coord(ax1, ax2, fig)

    if args.debug:
        core.mkdir_r('debug')
        with open('./debug/uv2xyz.txt', 'w') as f:
            f.write('pt1    pt2    pt3d\n')
            for i in range(len(pt1)):
                f.write('{p1}    {p2}    {p3d}\n'.format(
                    p1=pt1[i], p2=pt2[i], p3d=p3d[i]
                ))
    pass
