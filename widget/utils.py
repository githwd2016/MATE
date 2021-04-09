# -*- encoding: utf-8 -*-
"""
@File    : utils.py
@Time    : 2021/4/9 10:39
@Author  : Winton He
@Email   : hwd@mail.ustc.edu.cn
@Software: PyCharm
@Description:
"""
import pickle as pkl


def save_to_pkl(obj, pkl_file):
    # save to pkl_file
    print('save to {}...'.format(pkl_file))
    with open(pkl_file, 'wb') as f:
        pkl.dump(obj, f)
