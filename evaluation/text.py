# -*- coding: utf-8 -*-  
"""
@Author: winton
@File: main.py
@Time: 2019/6/17 9:39 AM
@Description:
"""
import argparse

from nltk.translate import bleu_score

from evaluation.nlgeval.pycocoevalcap.bleu.bleu_scorer import BleuScorer

score = 0


def main(args):
    with open(args.hypothesis, 'r', encoding='utf8') as f:
        hypothesis = [x.strip().split() for x in f.readlines()]
    references = []
    for reference in args.references:
        with open(reference, 'r', encoding='utf8') as f:
            references.append([x.strip().split() for x in f.readlines()])
    references = list(zip(*references))
    assert len(hypothesis) == len(references), 'length of hypothesis and references must be equal!'
    # corpus_bleu() is different from averaging sentence_bleu() for hypotheses
    print(bleu_score.corpus_bleu(references, hypothesis))
    # temp = 0
    # for reference, hyp in zip(references, hypothesis):
    #     temp += bleu_score.sentence_bleu(reference, hyp)
    # print(temp / len(hypothesis))
    # exit()

    with open(args.hypothesis, 'r', encoding='utf8') as f:
        hyp_list = f.readlines()
    ref_list = []
    for iidx, reference in enumerate(args.references):
        with open(reference, 'r', encoding='utf8') as f:
            ref_list.append(f.readlines())
    ref_list = [list(map(lambda s:s.strip(), refs)) for refs in zip(*ref_list)]
    refs = {idx: strippedlines for (idx, strippedlines) in enumerate(ref_list)}
    hyps = {idx: [lines.strip()] for (idx, lines) in enumerate(hyp_list)}

    assert (refs.keys() == hyps.keys())
    ids = refs.keys()
    bleu_scorer = BleuScorer(n=4)
    for _id in ids:
        hypo = hyps[_id]
        ref = refs[_id]
        # Sanity check.
        assert (type(hypo) is list)
        assert (len(hypo) == 1)
        assert (type(ref) is list)
        assert (len(ref) >= 1)
        bleu_scorer += (hypo[0], ref)
    score, scores = bleu_scorer.compute_score(option='closest', verbose=0)
    print(score)


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    _parser.add_argument('-ref', '--references', action='append', dest='references',
                         default=[], help='References file list.')
    _parser.add_argument('-hyp', '--hypothesis', help='Hypothesis file.')
    _args = _parser.parse_args()
    exit(main(_args))