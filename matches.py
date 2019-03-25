#!/usr/bin python3
# -*- coding: utf-8 -*-

'''批量匹配照片特征点
建议使用python -m　执行此脚本，跳过断言，提升执行速度
使用的命令将存入degub/matches-cmd.sh文件中

Examples:
----------
匹配参考照片: python3 matches.py -kpdir features/orb8000 -imgdir images/A 
-o matches/A --filter homography

匹配测试照片: python3 matches.py -kpdir features -imgdir images -o matches/T 
--filter homography --ifile ../gist_res.csv

Info:
----------
__require__: Python3.7及以上版本
__author__: devecor
'''

import argparse
import os
import time

import sys
sys.path.append(r'../')
import rename as ra
import const_params as const

def matches(kpfile1, kpfile2, img1, img2, output, Filter='homography'):
    start = time.time()
    start_cpu = time.thread_time()
    if Filter == 'None':
        Filter = ''
    else:
        Filter = '--' + Filter
    cmd = 'python3 check_match.py --kpfile1={kp1} --kpfile2={kp2} \
{filter} --save --output {output} {img1} {img2}'.format(
            kp1=kpfile1, kp2=kpfile2, img1=img1, img2=img2, 
            filter=Filter, output=output
        )
    os.system(cmd)

    escape_cpu = time.thread_time() - start_cpu
    escape = time.time() - start
    print('match {img1} and {img2} used: {t} seconds'.format(img1=img1, 
        img2=img2, t=escape))
    print('match {img1} and {img2} used cpu time: {t} seconds'.format(
        img1=img1, img2=img2, t=escape_cpu))
    return cmd

def get_kpfiles(imglist, kplist, config):
    '''组装参考点照片和参考点偏移照片的npz文件名
    Parameters
    -----------
    imglist : list
        findfile()的返回值
    kplist : list
        findfile()的返回值
    config : argparse的实例或者对象
        必须包含key_points_dir, images_dir两个属性
    '''
    kpf1 = []
    kpf2 = []
    img1 = []
    img2 = []
    args = config
    for i in imglist:
        image1 = i
        if sys.platform == 'linux':
            li = i.split('/')
        elif sys.platform == 'win32':
            li = i.split('\\')
        temp = list(li[-1])
        temp[0] = 'B'
        temp = ''.join(temp)
        li[-1] = temp
        image2 = '/'.join(li)

        img1.append(image1)
        img2.append(image2)

        kpfile1 = args.key_points_dir + \
            image1[len(args.images_dir):len(image1)-4] + \
            '{suffix}'.format(suffix=kplist[0][-8:])

        kpfile2 = args.key_points_dir + \
            image2[len(args.images_dir):len(image2)-4] + \
            '{suffix}'.format(suffix=kplist[0][-8:])
        
        kpf1.append(kpfile1)
        kpf2.append(kpfile2)
    return kpf1, kpf2, img1, img2

def csv2list(ifile,):
    import csv

    with open(ifile, newline='') as csvfile: 
        reader = csv.reader(csvfile)
        li = []
        for i in reader:
            li.append(i)
    
    for i,ei in enumerate(li): 
        for j,ej in enumerate(ei): 
            temp = ej.split(',') 
            temp[0] = temp[0][1:] 
            temp[-1] = temp[-1][:-1] 
            li[i][j] = temp
    
    li = [li[i] for i in range(len(li)) if i > 0]
    
    # 此段代码将str列表转换为float和int列表
    # 在元素级别上使用str()
    # 通过以上两步消掉了原str列表中的空格
    for i,ei in enumerate(li):
        for j,ej in enumerate(ei):
            for k,ek in enumerate(ej):
                if j > 1:
                    li[i][j][k] = str(float(ek))
                elif j >= 0 and j <= 1:
                    li[i][j][k] = str(int(ek))
                else:
                    print('不可能的事情发生了，这是鬼故事！')

    return li

def mk_dir(dir):
    path = dir.split('/')
    for i,e in enumerate(path):
        if isinstance(e, str):
            os.system('mkdir {dir}'.format(dir='/'.join(path[0:i+1])))

if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description='参考点照片的批量匹配')
    parser.add_argument('-kpdir', '--key_points_dir', 
        default='features', help='特征点文件的路径')
    parser.add_argument('-imgdir', '--images_dir', 
        default='./images', help='待匹配的照片目录')
    parser.add_argument('-o', '--output', default='matches', 
        help='结果保存路径')
    parser.add_argument('--filter', choices=['homography','fundamental', 'None'], 
        default='homography', help='选择过滤器')
    parser.add_argument('--ifile', default='None',
    help='如果指定了输入文件来限定匹配范围，则必须用于匹配测试照片')
    # parser.add_argument('-q', '--query', metavar='bool', choices=[True,False], 
    #     default=False, help='选择是否查询定位结果')
    args = parser.parse_args()

    mk_dir(args.output)

    debug = []

    if args.ifile is 'None':
        import sys
        if sys.platform == 'linux':
            key_points_dir = args.key_points_dir
            images_dir = args.images_dir
        elif sys.platform == 'win32':
            key_points_dir, images_dir = args.key_points_dir.replace('/', '\\'), args.images_dir.replace('/', '\\')
            # key_points_dir, images_dir = key_points_dir.replace('\\\\', '\\'), images_dir.replace('\\\\','\\')
        kplist = ra.findFile(key_points_dir, name='A*.npz')
        imglist = ra.findFile(images_dir, name='A*.jpg')

        begin = time.time()
        begin_cpu = time.thread_time()
        
        kpfile1,kpfile2,img1,img2 = get_kpfiles(imglist, kplist, args)
        if sys.platform == 'win32':
            for i, e in enumerate(kpfile1):
                kpfile1[i] = e.replace('/', '\\')

        for i in range(len(kpfile1)):
            
            assert kpfile1[i] in kplist, '凉凉'

            cmd = matches(kpfile1[i], kpfile2[i], img1[i], img2[i], args.output, args.filter)
            print(cmd)

        print('all used: {t} seconds'.format(t=time.time() - begin))
        print('all used cpu time: {t} seconds'.format(
            t=time.thread_time() - begin_cpu))

    elif os.path.splitext(args.ifile)[-1] == '.csv':
        
        li = csv2list(args.ifile)

        begin = time.time()
        begin_cpu = time.thread_time()
        
        for i in li:
            img1 = args.images_dir + '/T/T_' + '_'.join(i[0]) + '.jpg'
            img2 = args.images_dir + '/A/A_' + '_'.join(i[1]) + '.jpg'

            kpfile1 = args.key_points_dir + '/orb2000' + \
                img1[len(args.images_dir)+2:len(img1)-4] + \
                '{suffix}'.format(suffix='-orb.npz')

            kpfile2 = args.key_points_dir + '/orb8000' + \
                img2[len(args.images_dir)+2:len(img2)-4] + \
                '{suffix}'.format(suffix='-orb.npz')

            npzfiles = ra.findFile(args.key_points_dir, name='*.npz')
            assert kpfile1 in npzfiles and kpfile2 in npzfiles, '文件名有错误请检查！'
            
            cmd = matches(kpfile1, kpfile2, img1, img2, args.output, args.filter)
            debug.append(cmd)
        
        print('all used: {t} seconds'.format(t=time.time() - begin))
        print('all used cpu time: {t} seconds'.format(
            t=time.thread_time() - begin_cpu))
        
        with open('debug/matches-cmd.sh', 'w') as f:
            for i in debug:
                f.write(i + '\n')

    elif args.ifile == 'nogist':
        img_tp = ra.findFile(args.images_dir, name='T*.jpg')
        ref = []
        kpfile1 = []
        kpfile2 = []
        for i in img_tp:
            t = i.split('/')
            temp = t[-1].split('_')
            temp[-1] = temp[-1].split('.')[0]
            kp1 = temp.copy()
            temp[-1] = str(int() - 2)
            if temp[-1] == '-1':
                temp[-1] = '7'
            elif temp[-1] == '-2':
                temp[-1] = '8'

            temp[0] = 'A'
            for i in const.t2.correctPos.value:
                if i[0] == [int(temp[1]), int(temp[2])]:
                    temp[1], temp[2] = i[1]
                    temp[1], temp[2] = str(temp[1]), str(temp[2])
                    ref.append(args.images_dir + '/A/' + '_'.join(temp) + '.jpg')
                    kpfile2.append(args.key_points_dir + '/orb8000/' + '_'.join(temp) + '-orb.npz')
                    kpfile1.append(args.key_points_dir + '/orb2000/' + '_'.join(kp1) + '-orb.npz')
        
        with open('debug/query-list.txt', 'w') as f:
            for i in range(len(kpfile1)):
                f.write(str(kpfile1[i]) + ' ' + str(kpfile2[i]) + '\n')

        for i in range(len(img_tp)):

            npzfiles = ra.findFile(args.key_points_dir, name='*.npz')
            assert kpfile1[i] in npzfiles and kpfile2[i] in npzfiles, '文件名有错误请检查！'
            
            cmd = matches(kpfile1[i], kpfile2[i], img_tp[i], ref[i], args.filter, args.output)
            debug.append(cmd)
        with open('debug/matches-cmd.sh', 'w') as f:
            for i in debug:
                f.write(i + '\n')
        

    else:
        print('输入文件: {}　有误！'.format(args.ifile))