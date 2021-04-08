# -*- coding: utf-8 -*-  
"""
@Author: winton
@File: bleu_score.py
@Time: 2019/6/17 9:33 AM
@Description:
"""
from collections import Counter

import numpy as np
from nltk.translate import bleu_score


def bp(references_len, candidate_len):
    return np.e ** (1 - (candidate_len / references_len)) if references_len > candidate_len else 1


def nearest_len(references, candidate):
    return len(references[np.argmin([abs(len(i) - len(candidate)) for i in references])])


def parse_ngram(sentence, gram):
    return [sentence[i:i + gram] for i in range(len(sentence) - gram + 1)]


def appear_count(references, candidate, gram):
    # 对candidate和references分别分词（n-gram分词）
    ref = [parse_ngram(i, gram) for i in references]
    can = parse_ngram(candidate, gram)
    # 统计n-gram出现次数
    ref_counter = [Counter(i) for i in ref]
    can_counter = Counter(can)
    # 统计每个词在references中的出现次数
    # 对于candidate中的每个word，它的出现频次不能大于references中最大出现频次
    appear = sum(min(cnt, max(i.get(word, 0) for i in ref_counter)) for word, cnt in can_counter.items())
    return appear, len(can)


def corpus_bleu(references_list, candidate_list, weights):
    candidate_len = sum(len(i) for i in candidate_list)
    reference_len = sum(
        nearest_len(references, candidate) for candidate, references in zip(candidate_list, references_list))
    bp_value = bp(reference_len, candidate_len)
    s = 1
    for index, wei in enumerate(weights):
        up = 0  # 分子
        down = 0  # 分母
        gram = index + 1
        for candidate, references in zip(candidate_list, references_list):
            appear, total = appear_count(references, candidate, gram)
            up += appear
            down += total
        s *= (up / down) ** wei
    return bp_value * s


def sentence_bleu(references, candidate, weight):
    bp_value = bp(nearest_len(references, candidate), len(candidate))
    s = 1
    for gram, wei in enumerate(weight):
        gram = gram + 1
        appear, total = appear_count(references, candidate, gram)
        score = appear / total
        # 每个score的权值不一样
        s *= score ** wei
        # 最后的分数需要乘以惩罚因子
    return s * bp_value


if __name__ == '__main__':
    # ref1a = ["the", "dog", "jumps", "high"]
    # ref1b = ["the", "cat", "runs", "fast"]
    # ref1c = ["dog", "and", "cats", "are", "good", "friends"]
    # ref2a = ["ba", "ga", "ya"]
    # ref2b = ["lu", "ha", "a", "df"]
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b]]
    # hyp1 = ["the", "d", "o", "g", "jump", "s", "hig"]
    # hyp2 = ["it", "is", "too", "bad"]
    # candidate = [hyp1, hyp2]
    weights = [0.25, 0.25, 0.25, 0.25]
    references = [
        [
            "the dog jumps high",
            # "the cat runs fast",
            'dog and cats are good friends'
        ],
        [
            # "ba ga ya",
            'lu ha a df'
        ]
    ]
    candidate = ['the d o g  jump s hig', 'it is too bad']
    # print(corpus_bleu(references, candidate, weights))
    print(bleu_score.corpus_bleu(references, candidate, weights))
    # print(sentence_bleu(references[0], candidate[0], weights))
    # print(bleu_score.sentence_bleu(references[0], candidate[0], weights))
    print(bleu_score.sentence_bleu(['The color of the pants is black'.split()], 'The color of the pants is blue'.split(), weights))
