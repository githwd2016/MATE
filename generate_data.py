# -*- coding: utf-8 -*-  
"""
@Author: winton
@File: generate_data.py
@Time: 2019/7/9 11:12 AM
@Description:
"""
import argparse
import json
import os

from nltk import word_tokenize
from tqdm import tqdm

from widget.utils import save_to_pkl


def norm_sentence(sentence):
    # remove quotation marks and spaces at begin and end
    ret = sentence.lstrip('‘').rstrip('’').strip()
    # lower characters
    ret = ret.lower()
    # tokenize
    ret = ' '.join(word_tokenize(ret))
    return ret


def main(args):
    # process knowledge data
    knowledge_pairs = []
    with open(os.path.join(args.input_dir, 'styletips_synset.txt')) as file:
        for line in file:
            products = [None] * 2
            products[0], products[1], score = map(lambda x: x.strip(), line.split(','))
            products = list(map(lambda x: x.lower(), products))
            knowledge_pairs.append(products)
    with open(os.path.join(args.input_dir, 'celebrity_distribution.json')) as file:
        celebrity_json = json.load(file)
    for celebrity, products in celebrity_json.items():
        celebrity = celebrity.lower()
        for product in products.keys():
            product = product.lower()
            knowledge_pairs.append([celebrity, product])
    with open(os.path.join(args.out_dir, 'knowledge.json'), 'w', encoding='utf8') as file:
        json.dump(knowledge_pairs, file, indent=2, ensure_ascii=False)
    # process dialog data
    versions = ['v1', 'v2']
    splits = ['train', 'valid', 'test']
    for version in versions:
        for split in splits:
            path = os.path.join(args.input_dir, version, split)
            dialogs = []
            for file in tqdm(os.listdir(path), desc='Dump {} {}'.format(version, split)):
                with open(os.path.join(path, file), 'r') as f:
                    data = json.load(f)
                    dialog = []
                    for utterance in data:
                        # get utter attributes
                        speaker = utterance.get('speaker')
                        if 'question-subtype' in utterance:
                            utter_type = f"{utterance.get('type')}:{utterance.get('question-type')}:" \
                                         f"{utterance.get('question-subtype')}"
                        elif 'question-type' in utterance:
                            utter_type = f"{utterance.get('type')}:{utterance.get('question-type')}"
                        else:
                            utter_type = f"{utterance.get('type')}"
                        utter = utterance.get('utterance')
                        text = utter.get('nlg')
                        images = utter.get('images')
                        false_images = utter.get('false images')
                        # some attributes may be empty
                        if text is None:
                            text = ""
                        if images is None:
                            images = []
                        if false_images is None:
                            false_images = []
                        dialog.append((speaker, norm_sentence(text), images, false_images, utter_type))
                    dialogs.append(dialog)
            out_path = os.path.join(args.out_dir, version)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            out_file = os.path.join(out_path, f'{split}.pkl')
            save_to_pkl(dialogs, out_file)


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    # path
    _parser.add_argument('--input_dir', help='original data directory', required=True)
    _parser.add_argument('--out_dir', type=str, help='path for saving processed data', required=True)
    _args = _parser.parse_args()
    exit(main(_args))
