# -*- coding: utf-8 -*-
# @Time    : 18-6-27 上午10:19
# @Author  : zhoujun
import os
import pathlib
import cv2

def get_input_output_list(input_file_folder, output_file_folder, p_postfix=['.xml'], l_postfix='.txt',
                          save_to_same_folder=False):
    '''
    根据输入的路径是文件还是文件夹来产生输入和输出文件地址列表
    :param input_file_folder: 输入文件或文件夹地址
    :param output_file_folder: 输出文件或文件夹地址
    :param p_postfix: 原始文件后缀
    :param l_postfix: 目标文件后缀
    :return: 输入和输出文件列表
    '''
    input_p = pathlib.Path(input_file_folder)
    output_p = pathlib.Path(output_file_folder)
    if not input_p.exists():
        raise Exception('输入文件或目录不存在')

    if save_to_same_folder:
        i = 0
    if input_p.is_file():
        if save_to_same_folder:
            save_file = output_p.joinpath(str(i) + l_postfix)
        else:
            save_file = pathlib.Path(str(input_p).replace(input_file_folder, output_file_folder))
        if not save_file.parent.exists():
            os.makedirs(str(save_file.parent))
            print('makedir:%s', save_file.parent)
        yield input_p, save_file
    else:
        input_g_list = (input_p.rglob('*' + p) for p in p_postfix)
        for input_g in input_g_list:
            for input_path in input_g:
                if save_to_same_folder:
                    save_file = output_p.joinpath(str(i) + l_postfix)
                    i += 1
                else:
                    temp_file = pathlib.Path(str(input_path).replace(input_file_folder, output_file_folder))
                    save_file = pathlib.Path(temp_file.parent).joinpath(
                        temp_file.name.replace(input_path.suffix, l_postfix))
                if not save_file.parent.exists():
                    os.makedirs(str(save_file.parent))
                    print('makedir:%s', save_file.parent)
                yield input_path, save_file

def get_input_list(input_file_folder, p_postfix=['.xml']):
    '''
    根据输入的路径是文件还是文件夹来产生输入和输出文件地址列表
    :param input_file_folder: 输入文件或文件夹地址
    :param p_postfix: 原始文件后缀
    :return: 输入文件列表
    '''
    input_p = pathlib.Path(input_file_folder)
    if not input_p.exists():
        raise Exception('输入文件或目录不存在')

    if input_p.is_file():
        yield input_p
    else:
        input_g_list = (input_p.rglob('*' + p) for p in p_postfix)
        for input_g in input_g_list:
            for input_path in input_g:
                yield input_path

def rebuild(warp_img, mapx, mapy):
    return cv2.remap(warp_img, mapx, mapy, cv2.INTER_LINEAR)