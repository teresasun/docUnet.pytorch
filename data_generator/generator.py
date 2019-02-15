# -*- coding: utf-8 -*-
# @Time    : 2018/6/20 16:12
# @Author  : zhoujun
import numpy as np
import time
from ctypes import *
import multiprocessing as mp
import tqdm
import shutil
import numpy.ctypeslib as npct
from utils.utils import *
import argparse

array_2d_float = npct.ndpointer(np.float32, ndim=2, flags='C_CONTIGUOUS')


def load_dll(dll_path):
    return cdll.LoadLibrary(dll_path)


def clip_img_c(img_path, save_path):
    '''
    去除黑色背景
    :param img_path: 图像路径
    :param save_path: 图像保存路径
    :return:
    '''
    dll.clip_img(img_path, save_path)


def add_background(img_path, background_path, save_path, i):
    '''
    将扫描图片进行折叠或扭曲后和背景相结合以生成拍照图像
    :param img_path: 扫描图像路径
    :param background_path: 背景图像路径
    :param save_path: 图像保存路径
    :return:
    '''
    try:
        max_new_row = 600
        max_new_col = 800
        # img = cv2.imread(img_path)
        # 根据原图像计算新图像的宽高,新图像的宽高需要能被16整除 Unet需要这个
        # new_row = np.int32(np.ceil(img.shape[0] / 32)) * 16
        # new_col = np.int32(np.ceil(img.shape[1] / 32)) * 16
        # row, col = img.shape[:2]
        # if row > col:
        #     row_ratio = max_new_col / row
        #     col_ratio = max_new_row / col
        # else:
        #     row_ratio = max_new_row / row
        #     col_ratio = max_new_col / col
        # ratio = min(row_ratio, col_ratio)
        # new_row = int(row * ratio)
        # new_col = int(col * ratio)
        new_col = max_new_col
        new_row = max_new_row
        # 行方向的映射矩阵
        map_rows = np.ones((new_row, new_col), dtype=np.float32)
        # 列方向的映射矩阵, 和, vec_rows结合使用,
        # vec_rows[x] = x1, vec_rows[y] = y1
        #表示扫描图上(x, y)处的点在扭曲图上(x1, y1)处取值
        map_clos = np.ones((new_row, new_col), dtype=np.float32)
        np.random.seed(i)
        kernel = 9
        n = np.random.randint(1, 10)
        dll.generate_one_img_py(img_path.encode(), background_path.encode(), save_path.encode(), new_row, new_col,
                                map_rows, map_clos, n, i, kernel)
        np.save(save_path + '.npy', np.stack((map_clos, map_rows), axis=2))
    except Exception as e:
        print('error in {0}:{1}'.format(img_path, e))


def add_background_all(Flags):
    # img_path = '/data1/zj/docUnet/py/data_generator/1.jpg'  # '/data1/zj/data/src_img/22369.jpg'  # docUnet/py/data_generator/1.jpg'
    # back_path = '/data1/zj/docUnet/py/data_generator/bg.jpg'
    # save_path = '/data1/zj/docUnet/py/data_generator/1_result1.jpg'
    # start = time.time()
    # add_background(img_path, back_path, save_path, 2)
    # print(time.time() - start)

    input_file_folder = Flags.scan_images_path
    output_file_folder = Flags.background_images_path
    bg_dir = Flags.output_path
    all_num =  Flags.num

    g = get_input_output_list(input_file_folder, output_file_folder, ['.jpg', '.png'], '.jpg',save_to_same_folder=False)

    bg_list = list(get_input_list(bg_dir, ['.jpg', '.png']))

    print('start at', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    start = time.time()
    i = 0

    pool = mp.Pool(mp.cpu_count())
    pbar = tqdm.tqdm(total=all_num)
    for input_path, output_path in g:
        if input_path.exists() and not output_path.exists() and input_path.stat().st_size > 0:
            if not output_path.parent.exists():
                os.makedirs(str(output_path.parent))
            np.random.seed(i)
            idx = np.random.randint(0, len(bg_list))
            if not bg_list[idx].exists() or bg_list[idx].stat().st_size <= 0:
                continue
            # new_name = str(i) + output_path.suffix
            # output_path = output_path.parent / new_name
            # add_background(str(input_path), str(bg_list[idx]), str(output_path), i)
            pool.apply_async(func=add_background, args=(
                str(input_path), str(bg_list[idx]), str(output_path), i))
            i += 1
            pbar.update(1)
            if i >= all_num:
                break
    pbar.close()
    pool.close()
    pool.join()
    print(time.time() - start)


def remove_img(img_path, output_path, val=20):
    '''
    统计边界处的黑色像素值占所有像素的占比
    :param img_path: 扫描图像路径
    :param val: 阈值，三通道均低于这个值的被判断为黑色像素
    :return: 占比
    '''
    a = dll.remove_img(img_path.encode(), val)
    if a > 0.011:
        save_file = '%s_%0.4f.jpg' % (output_path[:-4], a)
        shutil.move(img_path, save_file)


def remove_img_thre(img_path, output_path):
    '''
    统计边界处的黑色像素值占所有像素的占比
    :param img_path: 扫描图像路径,图像会被二值化
    :return: 占比
    '''
    a = dll.remove_img_thre(img_path.encode())
    if a > 0.02:
        save_file = '%s_%0.4f.jpg' % (output_path[:-4], a)
        shutil.move(img_path, save_file)


def remove_img_all():
    dll.remove_img.restype = c_float
    dll.remove_img_thre.restype = c_float
    # img_path = r'D:\work\data\clip_black1\0\0_53.jpg'
    # save_path = r'D:\work\data\clip_black1\0\0_1.jpg'
    # remove_img(img_path.encode(), save_path.encode())
    # return
    input_file_folder = '/data1/zj/data/remove_img1'
    output_file_folder = '/data1/zj/data/remove_img2'
    input_list, output_list = get_input_output_list(
        input_file_folder, output_file_folder, ['.jpg', '.png'], '.jpg')
    assert len(input_list) == len(output_list)
    print('img numbers is ', len(input_list))

    pool = mp.Pool(mp.cpu_count() - 1)
    result = []
    print('start at', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    start = time.time()
    pbar = tqdm.tqdm(total=len(input_list))
    for input_path, output_path in zip(input_list, output_list):
        if os.path.exists(input_path) and not os.path.exists(output_path):
            save_dir = output_path.replace(output_path.split('/')[-1], '')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # result.append(pool.apply_async(func=remove_img, args=(input_path.encode(), 20)))
            pool.apply_async(func=remove_img_thre,
                             args=(input_path, output_path))
        pbar.update(1)
    pbar.close()
    pool.close()
    pool.join()
    print(time.time() - start)
    # result = [x.get() for x in result]
    # print(np.max(result))
    # print(np.mean(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--scan_images_path', type=str, default='',required=True,
                        help='the path of scan images')
    parser.add_argument('-b', '--background_images_path', type=str, default='', required=True,
                        help='the path of background images')
    parser.add_argument('-o', '--output_path', type=str, default='', required=True,
                        help='the path of output')
    parser.add_argument('-n', '--num', type=int, default=100000, required=True,
                        help='the number of generat images')
    Flags, _ = parser.parse_known_args()

    dll = npct.load_library("data_generator/cpp/add_background/build/libadd_background.so")
    dll.generate_one_img_py.restype = c_int
    dll.generate_one_img_py.argtypes = [c_char_p, c_char_p, c_char_p, c_int, c_int, array_2d_float, array_2d_float,
                                        c_int, c_int, c_int]
    add_background_all(Flags)

    # rebuild
#     start = time.time()
#     warp_img = cv2.imread('/data1/zj/docUnet/py/data_generator/1_result.jpg')
#     map_xy = np.load('/data1/zj/docUnet/py/data_generator/1_result.jpg.npy')
#     print(time.time() - start)
#     unwarp_img = rebuild(warp_img=warp_img, mapx=map_xy[:, :, 0], mapy=map_xy[:, :, 1])
#     print(time.time() - start)
#     cv2.imwrite('/data1/zj/docUnet/py/data_generator/1_result1.jpg', unwarp_img)
# #
