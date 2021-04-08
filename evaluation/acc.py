# -*- coding: utf-8 -*-  
"""
@Author: winton
@File: acc.py
@Time: 2019/9/10 3:52 PM
@Description:
"""
import argparse


def main(args):
    with open(args.hypothesis, 'r', encoding='utf8') as f:
        hypothesis = [x.strip().split() for x in f.readlines()]
    with open(args.references, 'r', encoding='utf8') as f:
        references = [x.strip().split() for x in f.readlines()]
    n_correct_words = 0
    n_words = 0
    for h, r in zip(hypothesis, references):
        for w1, w2 in zip(h, r):
            n_words += 1
            if w1 == w2:
                n_correct_words += 1
    print(f'Accuracy: {100 * n_correct_words / n_words:.5} %')


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    _parser.add_argument('-hyp', '--hypothesis', help='Hypothesis file.')
    _parser.add_argument('-ref', '--references', help='References file.')
    _args = _parser.parse_args()
    exit(main(_args))