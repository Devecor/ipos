#!/usr/bin python3
# -*- coding:utf-8 -*-

'''
使用orb匹配照片，使用tatio_test进行过滤，直接进行3维重建并可视化， 未进行
Info
------
__author__: devecor
'''
import core

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import argparse
import logging


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


def filter_match(kp1, kp2, matches, ratio=0.75, indirect=False):
    '''对匹配结果进行过滤, 将改变匹配结果
    parameters
    -----------
    kp1, kp2 : 关键点列表

    matches : 原始匹配结果, flannBasedMatch
    列表形式， 每个列表元素是一个DMatch实例

    ratio : 过滤条件, 默认为0.75

    indirect : 选择是否间接过滤

    return
    --------
    pt1, pt2 : 过滤后的关键点对应像素坐标序列

    kp_pairs : 正确匹配的关键点对列表

    mask : 表示是否正确匹配的掩码

    filtered_matches : 过滤之后的匹配结果
    '''
    mkp1, mkp2 = [], []
    mask1 = [0 for i in range(len(kp1))]
    mask1 = [mask1, mask1.copy()]
    mask2 = [[0, 0] for i in range(len(matches))]
    for i, m in enumerate(matches):
        if len(m) >= 2 and m[0].distance < m[1].distance * ratio:
            mkp1.append(kp1[m[0].queryIdx])
            mask1[0][m[0].queryIdx] = 1
            mkp2.append(kp2[m[0].trainIdx])
            mask1[1][m[0].trainIdx] = 1
            mask2[i] = [1, 0]
    pt1 = np.float64([kp.pt for kp in mkp1])
    pt2 = np.float64([kp.pt for kp in mkp2])
    kp_pairs = [mkp1, mkp2]
    filtered_matches = [e[0] for i, e in enumerate(matches)
                        if mask2[i] == [1, 0]]
    if indirect is False:
        return pt1, pt2, kp_pairs, mask2, filtered_matches
    else:
        return kp1, kp2, matches, mask1, mask2


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='detect_match')
    parser.add_argument('img', metavar='imagepair', nargs=2, help='一对照片')
    parser.add_argument('-nfeatures', default=8000, type=int, help='提取特征点数')
    parser.add_argument('--show', action='store_true', help='显示匹配结果')
    parser.add_argument('--save', action='store_true', help='是否保存结果')
    parser.add_argument('--output', metavar='dir', help='输出路径, 仅当指定--save时有效')
    parser.add_argument('--K1', metavar='float', help='内参矩阵：cx, cy, fx, fy')
    parser.add_argument('--K2', metavar='float', help='内参矩阵：cx, cy, fx, fy')
    parser.add_argument('--distCoeffs1', metavar='float', help='畸变参数: k1, k2, \
p1, p2')
    parser.add_argument('--distCoeffs2', metavar='float', help='畸变参数: k1, k2, \
p1, p2')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args(
        'images/A/s11-1.jpg images/A/s11-1a.jpg --show --save --output matches'.split()
        )
# --K 3123.8,3122.3,1497.6,2022.3
# ' --save --output matches \images/ref1.jpg images/ref2.jpg
# --K 3048,3048,1500,2000 --debug'.split()

    # 'images/A/s11-1.jpg images/A/s11-1a.jpg --save --output matches \
    # --K1 3178.2,3179.4,1497.5,2027.3 --K2 3130.4,3128.8,1487.6,2013.3 --debug \
    # --distCoeffs1 0.1692,-0.5744,0,0 \
    # --distCoeffs2 0.1520,-0.4518,0,0'.split()
    logging.basicConfig(level=logging.INFO)

    # 读取图片
    img1, img2 = args.img
    ref_img1 = cv.imread(img1, flags=cv.IMREAD_GRAYSCALE)
    ref_img2 = cv.imread(img2, flags=cv.IMREAD_GRAYSCALE)
    # tes_img1 = cv.imread('images/T/t20-3.jpg', flags=cv.IMREAD_GRAYSCALE)
    # tes_img2 = cv.imread('images/T/t20-4.jpg', flags=cv.IMREAD_GRAYSCALE)
    assert ref_img1 is not None and ref_img2 is not None, 'img not found!'

    # 特征提取与匹配
    matches, kp1, kp2, des1, des2 = match(ref_img1, ref_img2, args)
    logging.info('原始提取特征: {}'.format(len(kp1)))

    # matches_t, kp1_t, kp2_t, des1_t, des2_t = match(
    #     ref_img1, tes_img1, args
    # )

    # ratio test过滤
    pt1, pt2, kp_pairs, mask, filtered_matches = filter_match(
        kp1, kp2, matches.copy(), ratio=0.5)
    # pt1_t, pt2_t, kp_pairs_t, mask_t, filtered_matches_t = filter_match(
    #     kp1_t, kp2_t, matches_t.copy(), ratio=0.6
    # )
    logging.info('ratio test 过滤后: {}'.format(len(pt1)))

    # perspective过滤
    pt1, pt2, kp1, kp2, filtered_matches = core.filter_perspective(
        pt1, pt2, kp1=kp1, kp2=kp2,
        matches=filtered_matches, flags=2)
    logging.info('perspective过滤后: {}'.format(len(pt1)))

    if args.show:
        show_matches(ref_img1, kp1, ref_img2, kp2, filtered_matches)

    # 保存过滤结果到图片
    path = save_matches_to_img(ref_img1, kp1, ref_img2, kp2,
                               filtered_matches, config=args)

    # 保存过滤结果到文件
    np.save(path.split('.')[0], (pt1, pt2))
    logging.info('过滤结果保存到二进制文件: {}'.format(path.split('.')[0] + '.npy'))
