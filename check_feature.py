#!/usr/bin python3
# -*- coding: utf-8 -*-

import argparse
import glob
import json
import logging
import os
import sys

import itertools as it
from multiprocessing.pool import ThreadPool

import numpy as np
import cv2

import core

win = 'Detector'
asift_tilt = 3


def get_filenames(patterns):  # ipos未使用
    result = []
    for pat in patterns:
        result.extend(glob.glob(pat))
    return result


def save_keypoint_data(filename, camera, pose, keypoints, descriptors):
    kpdata = np.array([(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response,
                      kp.octave, kp.class_id) for kp in keypoints])
    np.savez(filename, camera=camera, pose=pose, keypoints=kpdata,
             descriptors=descriptors)
    return filename


def draw_str(dst, target, s):
    x, y = target
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 10.0, (0, 0, 0),
                thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 10.0, (255, 255, 255),
                lineType=cv2.LINE_AA)


def camera_matrix(config, img):
    h, w = img.shape[:2]
    if config.camera:
        fx, fy = [float(s) for s in config.camera.split(',')]
    else:
        fx, fy = 0, 0
    return np.float64([[fx*w, 0, 0.5*(w-1)],
                       [0, fy*h, 0.5*(h-1)],
                       [0.0, 0.0, 1.0]])


def explore_features(config, filename, img, kp, des):
    # draw only keypoints location, not size and orientation
    img2 = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
    logging.info('发现关键点总数目: %s', len(kp))

    if config.save:
        core.mkdir_r(config.output)  # add by devecor
        asift = 'asift.' if config.asift else ''
        destname = filename[:filename.rfind('.')] + '-' + asift + \
            config.feature
        if config.output:
            destname = os.path.join(config.output, os.path.basename(destname))

        logging.info('保存关键点到图片：%s', destname + '.jpg')
        cv2.imwrite(destname + '.jpg', cv2.pyrDown(img2))

        logging.info('保存关键点和描述符到文件：%s', destname + '.npz')
        camera = camera_matrix(config, img)
        if config.pose:
            pose = np.float64([float(s) for s in config.pose.split(',')])
        else:
            pose = np.zeros(5, dtype=np.float64)
        save_keypoint_data(destname, camera, pose, kp, des)

        logging.info('保存参数到文件：%s', destname + '.json')
        with open(destname + '.json', 'w') as f:
            data = dict(config._get_kwargs())
            data['filename'] = filename
            data['result'] = destname + '.npz'
            json.dump(data, f, indent=2)

    if config.show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.moveWindow(win, 0, 0)
        # h, w = img2.shape[:2]
        # s = 0.5
        # cv2.imshow(win, cv2.resize(img2, (int(w*s), int(h*s))))
        cv2.imshow(win, img2)
        cv2.waitKey()
        cv2.destroyWindow(win)


def affine_skew(tilt, phi, img, mask=None):
    '''
    affine_skew(tilt, phi, img, mask=None) -> skew_img, skew_mask, Ai

    Ai - is an affine transform matrix from skew_img to img
    '''
    h, w = img.shape[:2]
    if mask is None:
        mask = np.zeros((h, w), np.uint8)
        mask[:] = 255
    A = np.float32([[1, 0, 0], [0, 1, 0]])
    if phi != 0.0:
        phi = np.deg2rad(phi)
        s, c = np.sin(phi), np.cos(phi)
        A = np.float32([[c, -s], [s, c]])
        corners = [[0, 0], [w, 0], [w, h], [0, h]]
        tcorners = np.int32(np.dot(corners, A.T))
        x, y, w, h = cv2.boundingRect(tcorners.reshape(1, -1, 2))
        A = np.hstack([A, [[-x], [-y]]])
        img = cv2.warpAffine(img, A, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    if tilt != 1.0:
        s = 0.8*np.sqrt(tilt*tilt-1)
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=s, sigmaY=0.01)
        img = cv2.resize(img, (0, 0), fx=1.0/tilt, fy=1.0,
                         interpolation=cv2.INTER_NEAREST)
        A[0] /= tilt
    if phi != 0.0 or tilt != 1.0:
        h, w = img.shape[:2]
        mask = cv2.warpAffine(mask, A, (w, h), flags=cv2.INTER_NEAREST)
    Ai = cv2.invertAffineTransform(A)
    return img, mask, Ai


def affine_detect(detector, img, tilt, mask=None, pool=None):
    '''
    affine_detect(detector, img, mask=None, pool=None) -> keypoints, descrs

    Apply a set of affine transormations to the image, detect keypoints and
    reproject them into initial image coordinates.
    See http://www.ipol.im/pub/algo/my_affine_sift/ for the details.

    ThreadPool object may be passed to speedup the computation.
    '''
    params = [(1.0, 0.0)]
    delta = 72.0
    for t in 2**(0.5*np.arange(1, tilt)):
        for phi in np.arange(0, 180, delta / t):
            params.append((t, phi))
    logging.info('asift 取样参数: %s', params)

    def f(p):
        t, phi = p
        timg, tmask, Ai = affine_skew(t, phi, img, mask)
        keypoints, descrs = detector.detectAndCompute(timg, tmask)
        for kp in keypoints:
            x, y = kp.pt
            kp.pt = tuple(np.dot(Ai, (x, y, 1)))
        if descrs is None:
            descrs = []
        return keypoints, descrs

    keypoints, descrs = [], []
    if pool is None:
        ires = it.imap(f, params)
    else:
        ires = pool.imap(f, params)

    for i, (k, d) in enumerate(ires):
        # logging.debug('affine sampling: %d / %d' % (i+1, len(params)))
        keypoints.extend(k)
        descrs.extend(d)

    return keypoints, np.array(descrs)


def asift_detect_compute(config, detector, img, mask=None):
    pool = ThreadPool(processes=cv2.getNumberOfCPUs())
    return affine_detect(detector, img, config.tilt, mask=mask, pool=pool)


def asift_query_feature(config):
    logging.info('启用 asift ')
    logging.info('asift_tilt is %s', config.tilt)

    detector = create_detector(config)
    for filename in core.get_filenames(config.images, name='.jpg'):
        logging.info('当前图片: %s', filename)
        img = cv2.imread(filename, 0)

        if config.grid:
            kp, des = [], []
            h, w = img.shape[:2]
            rows, cols = 3, 3
            cells = get_cells(h, w, rows, cols)
            logging.info("使用九宫格搜索特征: %s 行, %s 列", rows, cols)
            for x0, y0, x1, y1 in cells:
                logging.info("正在搜索九宫格: %s", (x0, y0, x1, y1))
                mask = np.zeros((h, w), dtype=np.uint8)
                mask[y0:y1, x0:x1] = 255
                features = asift_detect_compute(config, detector, img, mask)
                if features[0] is not None and features[1] is not None:
                    kp.extend(features[0])
                    des.extend(features[1])
        else:
            if config.mask:
                h, w = img.shape[:2]
                x0, y0, x1, y1 = [int(s) for s in config.mask.split(',')]
                mask = np.zeros((h, w), dtype=np.uint8)
                mask[y0:y1, x0:x1] = 255
            else:
                mask = None
            kp, des = asift_detect_compute(config, detector, img, mask)
        explore_features(config, filename, img, kp, des)


def get_cells(h, w, row=3, col=3):
    rows = list(range(0, h + 1, h / row))
    rows[-1] = h - 1
    cols = list(range(0, w + 1, w / col))
    cols[-1] = w - 1
    cells = []
    for i in range(row):
        for j in range(col):
            x0, y0, x1, y1 = cols[j], rows[i], cols[j+1], rows[i+1]
            cells.append((x0, y0, x1, y1))
    return cells


def query_feature(config):
    detector = create_detector(config)
    for filename in core.get_filenames(config.images, name='*.jpg'):
        logging.info('当前图片: %s', filename)
        img = cv2.imread(filename, 0)
        # kp = detector.detect(img, mask)
        # kp, des = detector.compute(img, kp)
        if config.grid:
            kp, des = [], []
            h, w = img.shape[:2]
            rows, cols = 3, 3
            cells = get_cells(h, w, rows, cols)
            logging.info("使用九宫格搜索特征: %s 行, %s 列", rows, cols)
            for x0, y0, x1, y1 in cells:
                logging.info("正在搜索九宫格: %s", (x0, y0, x1, y1))
                mask = np.zeros((h, w), dtype=np.uint8)
                mask[y0:y1, x0:x1] = 255
                features = detector.detectAndCompute(img, mask)
                if features[0] is not None and features[1] is not None:
                    kp.extend(features[0])
                    des.extend(features[1])
        else:
            if config.mask:
                h, w = img.shape[:2]
                x0, y0, x1, y1 = [int(s) for s in config.mask.split(',')]
                mask = np.zeros((h, w), dtype=np.uint8)
                mask[y0:y1, x0:x1] = 255
            else:
                mask = None
            kp, des = detector.detectAndCompute(img, mask)
        explore_features(config, filename, img, kp, des)


def grid_query_feature(config):
    if config.mask:
        logging.warn("使用九宫格进行特征搜索，命令行参数 --mask=%s 被忽略", config.mask)


def create_detector(config):
    if config.feature == 'akaze':
        detector = cv2.AKAZE_create()
    elif config.feature == 'sift':
        logging.info('使用 sift 算法查询关键点')
        logging.info('nFeatures = %d', config.nFeatures)
        detector = cv2.xfeatures2d.SIFT_create(config.nFeatures)
    elif config.feature == 'surf':
        logging.info('使用 surf 算法查询关键点')
        detector = cv2.xfeatures2d.SURF_create(400)
    else:
        logging.info('使用 orb 算法查询关键点')
        logging.info('nFeatures = %d', config.nFeatures)
        detector = cv2.ORB_create(config.nFeatures)
    return detector


def main():
    parser = argparse.ArgumentParser(description='查看图片特征和关键点')
    parser.add_argument('images', metavar='FILEDIR', help='图片文件目录')
    # parser.add_argument('-c', metavar='CONFIG.JSON', help='特征参数配置文件')
    parser.add_argument('--show', action='store_true', help='在窗口中显示包含关键点的图片')
    parser.add_argument('--save', action='store_true', help='保存包含关键点的图片')
    parser.add_argument('--output', help='输出文件的路径')
    parser.add_argument('--grid', action='store_true', help='使用九宫格获取关键点')
    parser.add_argument('--mask', help='选择区域（x0, y0, x1, y1)')
    parser.add_argument('--asift', action='store_true', help='使用asift算法')
    parser.add_argument('--tilt', type=int, default=asift_tilt,
                        help='设置 asift 的 tilt 参数')
    parser.add_argument('--feature', choices=['orb', 'sift', 'surf'],
                        default='orb', help='特征名称')
    parser.add_argument('--nFeatures', metavar='n', type=int,
                        default=800, help='特征数目')
    parser.add_argument('--pose', help='参考平面和相机距离，水平方位角和相机位置(d,a,x,y,z)')
    parser.add_argument('--camera', help='相机内参(fx,fy)')
    args = parser.parse_args()

    if args.asift:
        asift_query_feature(args)
    else:
        query_feature(args)

    if args.show:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    main()
