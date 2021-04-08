# -*- coding: utf-8 -*-  
"""
@Author: winton
@File: test.py
@Time: 2019/7/29 10:49 AM
@Description:
"""
from nlgeval import compute_metrics, compute_individual_metrics
from nltk.translate import bleu_score

reference = [
    ['The', 'new', 'translator', 'will', 'stand', 'on', 'the', 'exhibition', 'on', 'behalf', 'of', 'the', 'four',
     'times', 'group', 'at', 'the', 'exhibition', 'We', 'will', 'introduce', 'the', 'new', 'star`s', 'business', 'the',
     'advantages', 'and', 'the', 'successful', 'cases', 'so', 'that', 'you', 'can', 'understand', 'the', 'new',
     'translator', 'more', 'comprehensively', 'We', 'have', 'a', 'stable', 'full-time', 'international', 'team', 'that',
     'ensures', 'punctual', 'efficient', 'translation', 'and', 'dubbing', 'and', 'provides', 'a', 'full', 'range', 'of',
     'control', 'through', 'the', 'perfect', 'quality', 'control', 'and', 'project', 'management', 'system',
     'providing', 'a', 'one-stop', 'service', 'for', 'translation', 'dubbing', 'subtitle', 'production', 'post',
     'production', 'broadcasting', 'and', 'ratings', 'surveys'],
    ['The', 'new', 'translator', 'star', 'will', 'represent', 'sida', 'times', 'group', 'in', 'the', 'exhibition',
     'when', 'we', 'will', 'introduce', 'the', 'new', 'translator', 'star`s', 'business', 'advantages', 'successful',
     'cases', 'and', 'other', 'dimensions', 'so', 'that', 'you', 'can', 'have', 'a', 'more', 'comprehensive',
     'understanding', 'of', 'the', 'new', 'translator', 'star', 'We', 'have', 'a', 'stable', 'full-time',
     'international', 'team', 'which', 'can', 'ensure', 'timely', 'and', 'efficient', 'translation', 'and', 'dubbing',
     'Through', 'perfect', 'quality', 'control', 'and', 'project', 'management', 'system', 'we', 'provide',
     'translation', 'dubbing', 'subtitle', 'production', 'post-production', 'broadcasting', 'and', 'rating', 'survey']]
candidate = ['New', 'Transtar', 'will', 'present', 'itself', 'at', 'the', 'Exhibition', 'on', 'behalf', 'of',
             'StarTimes', 'and', 'we', 'will', 'give', 'a', 'comprehensive', 'introduction', 'of', 'ourselves',
             'including', 'the', 'current', 'services', 'we', 'offer', 'the', 'advantages', 'we', 'hold', 'and', 'the',
             'projects', 'we', 'have', 'completed', 'to', 'help', 'you', 'understand', 'us', 'more', 'New', 'Transtar',
             'boasts', 'of', 'an', 'international', 'team', 'of', 'professionals', 'and', 'is', 'capable', 'of',
             'providing', 'fast', 'and', 'quality-guaranteed', 'services', 'including', 'translating', 'dubbing',
             'subtitle', 'making', 'post-production', 'broadcasting', 'and', 'collecting', 'of', 'viewership',
             'ratings', 'thanks', 'to', 'our', 'strict', 'streamlined', 'and', 'developed', 'quality', 'control', 'and',
             'project', 'management', 'system']
metrics_dict = compute_individual_metrics([' '.join(ref) for ref in reference], ' '.join(candidate))
sentence_bleu = bleu_score.sentence_bleu(reference, candidate)
print(metrics_dict)
print(sentence_bleu)
hypothesis_path = '/home/hwd/project/mmkbds/model/baseline/cmmd/result/cmmd_v4/pred_10.txt'
references_path = '/home/hwd/dataset/cmmd/data/dataset/v2/dialogue_data/context_2_20/test_target_text.txt'
metrics_dict = compute_metrics(hypothesis=hypothesis_path, references=[references_path])
with open(hypothesis_path, 'r') as f:
    hypothesis = [x.strip().split() for x in f.readlines()]
references = []
for reference in references_path:
    with open(reference, 'r') as f:
        references.append([x.strip().split() for x in f.readlines()])
references = list(zip(*references))
corpus_bleu = bleu_score.corpus_bleu(references, hypothesis)
print(metrics_dict)
print(corpus_bleu)