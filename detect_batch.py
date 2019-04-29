#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''特征提取与保存批处理'''

import logging
import argparse
import glob

import detect_match as dm

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='特征提取保存批处理')
    parser.add_argument('imgs', metavar='dir', help='图片集目录')
    parser.add_argument('-o', '--output', metavar='dir', help='输出目录, 若不存在将被创建')
    parser.add_argument('-f', '--feature', choices=['orb', 'sift'], help='何种特征')
    parser.add_argument('-n', '--nfearure', type=int, help='待提取的特征数')
    args = parser.parse_args(
        'images/*.jpg'.split()
    )

    img_li = glob.glob(args.imgs)
    assert len(img_li) > 0, 'imgs are not found'
    for i in img_li:
        kp, des, img = dm.detect(i, feature=args.feature, n=args.nfeature)
    pass
