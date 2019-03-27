# -*- coding: utf-8 -*-
#

import argparse
import json
import logging
import os
import sys

import numpy as np
import cv2

FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH = 6

win = 'Matcher '


def load_keypoint_data(filename):
    npzfile = np.load(filename if filename.endswith('.npz') else (filename + '.npz'))
    keypoints = [cv2.KeyPoint(x, y, size, angle, response, int(octave), int(class_id))
                 for x, y, size, angle, response, octave, class_id in npzfile['keypoints']]
    return keypoints, npzfile['descriptors']


def draw_str(dst, target, s):
    x, y = target
    cv2.putText(dst, s, (x+10, y+10), cv2.FONT_HERSHEY_PLAIN, 10.0, (0, 255, 0), thickness = 10, lineType=cv2.FILLED)
    # cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 10.0, (255, 255, 255), lineType=cv2.FILLED)


def explore_match(config, kp_pairs, status = None, H = None):
    img1 = cv2.imread(config.filenames[0], 0)
    img2 = cv2.imread(config.filenames[1], 0)
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if H is not None:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32(cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0))
        cv2.polylines(vis, [corners], True, (255, 255, 255))

    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)

    p1, p2 = [], []  # python 2 / python 3 change of zip unpacking
    for kpp in kp_pairs:
        p1.append(np.int32(kpp[0].pt))
        p2.append(np.int32(np.array(kpp[1].pt) + [w1, 0]))

    green = (0, 255, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)
    kp_color = (51, 103, 236)
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            col = green
            cv2.circle(vis, (x1, y1), 2, col, -1)
            cv2.circle(vis, (x2, y2), 2, col, -1)
        else:
            col = red
            r = 2
            thickness = 3
            cv2.line(vis, (x1-r, y1-r), (x1+r, y1+r), col, thickness)
            cv2.line(vis, (x1-r, y1+r), (x1+r, y1-r), col, thickness)
            cv2.line(vis, (x2-r, y2-r), (x2+r, y2+r), col, thickness)
            cv2.line(vis, (x2-r, y2+r), (x2+r, y2-r), col, thickness)

    vis0 = vis.copy()
    matched = []
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv2.line(vis, (x1, y1), (x2, y2), green, 3)
            matched.append((x2, y2))
    n = np.count_nonzero(status)
    n0 = len(set(matched))
    logging.info('匹配的结果(unique/inlier/outlier)： %s / %s / %s', n0, n,
                 len(kp_pairs))
    draw_str(vis, (100, 200), 'unique / inliner / outliner: %s / %s / %s' % (
        n0, n, len(kp_pairs)))

    if config.save:
        import core
        core.mkdir_r(config.output)
        filename = '%s-%s.jpg' % (os.path.basename(config.filenames[0]).rsplit(
            '.')[0], os.path.basename(config.filenames[1]).rsplit('.')[0])
        if config.output:
            filename = os.path.join(config.output, os.path.basename(filename))
        logging.info('保存匹配结果到图片：%s', filename)
        cv2.imwrite(filename, cv2.pyrDown(vis));
        filename = os.path.splitext(filename)[0] + '.match.txt'
        logging.info('保存匹配数据到文件：%s', filename)
        with open(filename, 'w') as f:
            f.write('%-10s %-10s %-10s %-10s %-10s\n' % (os.path.basename(
                config.filenames[0]), os.path.basename(config.filenames[1]),
                n0, n, len(kp_pairs)))

    if not config.show:
        return

    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.moveWindow(win, 0, 0)
    cv2.imshow(win, vis)

    def onmouse(event, x, y, flags, param):
        def anorm2(a):
            return (a*a).sum(-1)

        def anorm(a):
            return np.sqrt( anorm2(a) )
        cur_vis = vis
        if flags & cv2.EVENT_FLAG_LBUTTON:
            cur_vis = vis0.copy()
            r = 8
            m = (anorm(np.array(p1) - (x, y)) < r) | (anorm(np.array(p2) - (x, y)) < r)
            idxs = np.where(m)[0]
            kp1s, kp2s = [], []
            for i in idxs:
                (x1, y1), (x2, y2) = p1[i], p2[i]
                col = (red, green)[status[i]]
                cv2.line(cur_vis, (x1, y1), (x2, y2), col)
                kp1, kp2 = kp_pairs[i]
                kp1s.append(kp1)
                kp2s.append(kp2)
            cur_vis = cv2.drawKeypoints(cur_vis, kp1s, None, flags=4,
                                        color=kp_color)
            cur_vis[:, w1:] = cv2.drawKeypoints(cur_vis[:, w1:], kp2s,
                                                None, flags=4, color=kp_color)

        cv2.imshow(win, cur_vis)
    cv2.setMouseCallback(win, onmouse)
    cv2.waitKey()
    cv2.destroyWindow(win)
    return vis


def filter_matches(kp1, kp2, matches, ratio=0.75):
    mkp1, mkp2 = [], []
    mi = []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append(kp1[m.queryIdx])
            mkp2.append(kp2[m.trainIdx])
            mi.append(m)
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = list(zip(mkp1, mkp2))
    return mi, p1, p2, list(kp_pairs)


def match_features(config, kp1, desc1, kp2, desc2):

    if config.kdtree:
        flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    else:
        flann_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,  # 12
                            key_size=12,     # 20
                            multi_probe_level=1)  # 2
    matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)

    raw_matches = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)  # 2
    logging.info('原始匹配数目： %s', len(raw_matches))

    mi, p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)
    logging.info('过滤之后匹配数目： %s', len(kp_pairs))
    # p1 = cv2.KeyPoint_convert(kp_pairs[0])
    # p1 = cv2.KeyPoint_convert(kp_pairs[1])

    # H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
    # F, status = cv2.findFundamentalMat(p1, p2)
    # H, status = cv2.findHomography(p1, p2, 0)

    H, status = None, None

    if len(p1) >= 4:
        if config.homography:
            H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 3.0)
        elif config.fundamental:
            H, status = cv2.findFundamentalMat(p1, p2)
        # do not draw outliers (there will be a lot of them)
        # kp_pairs = [kpp for kpp, flag in zip(kp_pairs, status) if flag]
        # mp = [dma.queryIdx for dma, flag in zip(mi, status) if flag]
        # logging.info('看看这是什么: %s, %s', len(kp_pairs), len(status))
        # return mp, status, kp_pairs

    explore_match(config, kp_pairs, status, H)


def main(params=None):
    parser = argparse.ArgumentParser(description='匹配两个图片的关键点')
    parser.add_argument('filenames', metavar='FILENAME', nargs=2, help='训练图片，查询图片')
    parser.add_argument('--kpfile1', help='训练图片关键点文件')
    parser.add_argument('--kpfile2', help='查询图片关键点文件')
    parser.add_argument('--path', help='关键点文件的路径')
    parser.add_argument('--suffix', default='orb', help='关键点文件的后缀名称')
    parser.add_argument('--kdtree', action='store_true',
                        help='使用 FLANN_INDEX_KDTREE 进行匹配')
    parser.add_argument('--homography', action='store_true',
                        help='是否使用 homography 过滤匹配结果')
    parser.add_argument('--fundamental', action='store_true',
                        help='是否使用 fundamental 过滤匹配结果')
    parser.add_argument('--show', action='store_true', help='在窗口中显示匹配结果')
    parser.add_argument('--save', action='store_true', help='是否保存匹配结果')
    parser.add_argument('--output', help='输出文件的路径')
    args = parser.parse_args('--kpfile1=features/orb2000/t20-3-orb.npz --kpfile2=features/orb8000/s11-1-orb.npz --homography --save --output matches\T images/T/t20-3.jpg images/A/s11-1.jpg'.split())

    filename1, filename2 = args.kpfile1, args.kpfile2
    if filename1 is None or filename2 is None:
        if ',' in args.suffix:
            suffix = ['-' + x for x in args.suffix.split(',')]
        else:
            suffix = '-' + args.suffix
            suffix = [suffix, suffix]
        filename1 = os.path.join(args.path, os.path.basename(args.filenames[0]).rsplit('.')[0] + suffix[0])
        filename2 = os.path.join(args.path, os.path.basename(args.filenames[1]).rsplit('.')[0] + suffix[1])
    logging.info("%s, %s", filename1, filename2)
    # 从本地文件中获取关键点和描述符
    kp1, des1 = load_keypoint_data(filename1)
    kp2, des2 = load_keypoint_data(filename2)

    match_features(args, kp1, des1, kp2, des2)

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    main()
