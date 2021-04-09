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
            out_file = os.path.join(args.out_dir, version, f'{split}.pkl')
            save_to_pkl(dialogs, out_file)


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    # path
    _parser.add_argument('--input_dir', help='original data directory', required=True)
    _parser.add_argument('--out_dir', type=str, help='path for saving processed data', required=True)
    _args = _parser.parse_args()
    exit(main(_args))
