# -*- coding: utf-8 -*-  
"""
@Author: winton
@File: vis.py
@Time: 2019/9/16 10:09 AM
@Description:
"""
import argparse
import re
import matplotlib.pyplot as plt


def main(args):
    train_pattern = re.compile(r'INFO: Epoch \[\d*\] Train Loss: (\d*\.\d*), '
                               r'Train Perplexity: (\d*\.\d*), Train Accuracy: (\d*\.\d*) %')
    valid_pattern = re.compile(r'INFO: Epoch \[\d*\] Valid Loss: (\d*\.\d*), '
                               r'Valid Perplexity: (\d*\.\d*), Valid Accuracy: (\d*\.\d*) %')
    train_loss = []
    train_perplexity = []
    train_accuracy = []
    valid_loss = []
    valid_perplexity = []
    valid_accuracy = []
    with open(args.file, 'r') as f:
        for line in f.readlines():
            match = train_pattern.search(line.strip())
            if match:
                groups = match.groups()
                train_loss.append(float(groups[0]))
                train_perplexity.append(float(groups[1]))
                train_accuracy.append(float(groups[2]))
            match = valid_pattern.search(line.strip())
            if match:
                groups = match.groups()
                valid_loss.append(float(groups[0]))
                valid_perplexity.append(float(groups[1]))
                valid_accuracy.append(float(groups[2]))
    epoch = list(range(1, len(train_loss) + 1))
    # print(epoch)
    # print(train_loss)
    # print(valid_loss)
    # exit()
    fig = plt.figure()
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax1.plot(epoch, train_loss, marker='o', c='red', label='train')
    ax1.plot(epoch, valid_loss, marker='o', c='blue', label='valid')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='best')
    ax2.plot(epoch, train_perplexity, marker='o', c='red', label='train')
    ax2.plot(epoch, valid_perplexity, marker='o', c='blue', label='valid')
    ax2.set_title('Perplexity')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Perplexity')
    ax2.legend(loc='best')
    ax3.plot(epoch, train_accuracy, marker='o', c='red', label='train')
    ax3.plot(epoch, valid_accuracy, marker='o', c='blue', label='valid')
    ax3.set_title('Accuracy')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    _parser.add_argument('--file', required=True)
    _args = _parser.parse_args()
    exit(main(_args))
