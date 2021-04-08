# -*- coding: utf-8 -*-  
"""
@Author: winton
@File: recall.py
@Time: 2019/8/9 9:19 AM
@Description:
"""
import argparse


def main(args):
    total_recall = [0, 0, 0]
    count = 0
    if args.metric == 'recall':
        with open(args.out_file, 'r') as f:
            for line in f.readlines():
                count += 1
                temp = line.split('\t')
                true_temp = [float(i) for i in temp[0].split()]
                false_temp = [float(i) for i in temp[1].split()]
                label = [1] * len(true_temp) + [0] * len(false_temp)
                result = list(sorted(zip(label, true_temp + false_temp), key=lambda d: d[1], reverse=True))
                result = list(zip(*result))[0]
                recall = [sum(result[:i]) / len(true_temp) for i in range(1, len(result) + 1)]
                for i in range(min(len(total_recall), len(recall))):
                    total_recall[i] += recall[i]
            for i, r in enumerate(total_recall, 1):
                print(f'Recall@{i}: {r / count}')


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    _parser.add_argument('--out_file', type=str, help='path for saving result', required=True)
    _parser.add_argument('--metric', type=str, help='metric for evaluation', choices=['recall', 'ndcg'], required=True)
    _args = _parser.parse_args()
    exit(main(_args))
