# -*- coding: utf-8 -*-  
"""
@Author: winton
@File: data.py
@Time: 2021/4/9 3:59 PM
@Description:
"""
import json
import os
from os.path import isfile
from collections import Counter

import pickle as pkl

import numpy as np
from nltk import word_tokenize
from torch.utils.data import Dataset
from annoy import AnnoyIndex
from tqdm import tqdm

from widget.utils import save_to_pkl

PAD_ID = 0
UNK_ID = 1
START_ID = 2
END_ID = 3


def pad_text(vocab, length, text):
    text = list(map(lambda w: vocab.get(w.lower(), UNK_ID), word_tokenize(text)))
    if len(text) > length - 1:
        text = text[:length - 1]
        text.append(END_ID)
        text_length = length
    else:
        text_length = len(text) + 1
        text.append(END_ID)
        text.extend([PAD_ID for _ in range(length - len(text))])
    return text, text_length


def pad_img(length, imgs):
    if len(imgs) > length:
        imgs = imgs[:length]
        imgs_length = length
    else:
        imgs_length = len(imgs)
        imgs.extend(['' for _ in range(length - len(imgs))])
    return imgs, imgs_length


def save_to_txt(texts, file):
    with open(file, 'w') as f:
        for text in texts:
            f.write(text + '\n')


class DataSource(Dataset):
    def __init__(self, config_path, task, mode, version, context_size):
        assert task in ['text', 'image']
        assert mode in ['train', 'valid', 'test']
        # Load config
        self.config = json.load(open(config_path, 'r'))
        self.task = task
        self.mode = mode
        self.context_size = context_size
        self.dialogs = None
        self.source_file = os.path.join(self.config['data']['source_path'], f'v{version}', f'{mode}.pkl')
        work_path = os.path.join(self.config['data']['work_path'], f'v{version}_c{context_size}')
        if not os.path.exists(work_path):
            os.makedirs(work_path)
        item_file = os.path.join(work_path, f'{task}_{mode}_item.pkl')
        gt_file = os.path.join(work_path, f'gt_text.txt')
        type_file = os.path.join(work_path, f'dialog_type.txt')
        self.vocab_file = os.path.join(work_path, f'vocab.pkl')
        if not isfile(self.vocab_file) or not isfile(item_file):
            print('reading dialog pkl...')
            self.dialogs = pkl.load(open(self.source_file, 'rb'))
        self.image_pos = ['1st', '2nd', '3rd', '4th', '5th', '6th',
                          '7th', '8th', '9th', '10th', '11th', '12th'][:self.config['data']['image_length']]
        self.vocab = self.create_or_load_vocab()
        if 'annoy_file' in self.config['data']:
            # use pre-train VGG image vector
            self.annoy = AnnoyIndex(self.config['model']['image_size'], metric='euclidean')
            self.annoy.load(self.config['data']['annoy_file'])
            self.annoy_index = pkl.load(open(self.config['data']['annoy_pkl'], 'rb'))
        if not isfile(item_file):
            self.items, gt_texts, dialog_types = self.get_items_from_dialogs()
            save_to_pkl(self.items, item_file)
            if task == 'text' and mode == 'test':
                save_to_txt(gt_texts, gt_file)
                save_to_txt(dialog_types, type_file)
        else:
            print(f'reading {task} {mode} item pkl...')
            self.items = pkl.load(open(item_file, 'rb'))
            print(f'{task} {mode} item pkl read complete')

    def create_or_load_vocab(self):
        if not isfile(self.vocab_file):
            if self.mode == 'train':
                # get vocab
                print('get vocab...')
                word_freq = Counter()
                for dialog in self.dialogs:
                    for utterance in dialog:
                        # (speaker, text, images, false_images, utter_type)
                        word_freq.update([word for word in word_tokenize(utterance[1])])
                words = [word for word, freq in word_freq.most_common()
                         if freq >= self.config['data']['context_text_cutoff']]
                vocab = {word: wid for wid, word in enumerate(words, 4)}
                vocab['<pad>'] = PAD_ID
                vocab['<unk>'] = UNK_ID
                vocab['</s>'] = START_ID
                vocab['</e>'] = END_ID
                # update speaker, turn
                assert '@user' not in vocab, '@user already exist!'
                vocab['@user'] = len(vocab)
                assert '@system' not in vocab, '@user already exist!'
                vocab['@system'] = len(vocab)
                for i in range(self.context_size):
                    assert f'#{i}' not in vocab, f'#{i} already exist!'
                    vocab[f'#{i}'] = len(vocab)
                for pos_i in self.image_pos:
                    if pos_i not in vocab:
                        print(f'{pos_i} not in vocabulary')
                        vocab[pos_i] = len(vocab)
                save_to_pkl(vocab, self.vocab_file)
            else:
                raise Exception('Vocabulary is not exist!')
        else:
            print('reading vocab pkl...')
            vocab = pkl.load(open(self.vocab_file, 'rb'))
            print('vocab pkl read complete')
        return vocab

    def get_items_from_dialogs(self):
        empty_text = [END_ID] + [PAD_ID] * (self.config['data']['text_length'] - 1)
        prog = tqdm(total=len(self.dialogs), desc='get items from dialogs')
        items = []
        gt_texts = []
        dialog_types = []
        for dialog in self.dialogs:
            history = [(empty_text, 1, [''] * self.config['data']['image_length'], 0, '<pad>', '<pad>')] * self.context_size
            for utter in dialog:
                # (speaker, text, images, false_images, utter_type)
                text, text_length = pad_text(self.vocab, self.config['data']['text_length'], utter[1])
                img, img_length = pad_img(self.config['data']['image_length'], utter[2])
                if utter[0] == 'user':
                    history.append((text, text_length, img, img_length, '@user', utter[4]))
                else:
                    context = history[-self.context_size:]
                    if self.task == 'text':
                        if len(utter[1]) > 0:
                            items.append((context, (text, text_length)))
                            gt_texts.append(utter[1])
                            dialog_types.append(history[-1][5])
                    else:
                        # true_images = self.get_img_urls(utter.images)
                        # false_images = self.get_img_urls(utter.false_images)
                        # # ensure images number enough
                        # if len(true_images) < 1 or len(false_images) < 1:
                        #     continue
                        # true_images, true_images_num = pad_img(self.config['data']['num_pos_images'], true_images)
                        # false_images, false_images_num = pad_img(self.config['data']['num_neg_images'], false_images)
                        # self.items.append((context, (true_images, true_images_num, false_images, false_images_num)))
                        pass
                    history.append((text, text_length, img, img_length, '@system', utter[4]))
            prog.update()
        prog.close()
        return items, gt_texts, dialog_types

    def __getitem__(self, index):
        item = self.items[index % len(self.items)]
        # (text, text_length, img, img_length, speaker, dialog_type)
        texts, text_lengths, imgs, img_lengths, speakers, _ = map(list, zip(*item[0]))
        text_input = []
        text_pos = []
        text_turn = []
        text_speaker = []
        for turn, (text, length, speaker) in enumerate(zip(texts, text_lengths, speakers)):
            text_input.extend(text)
            text_pos.extend(list(range(1, length + 1)) + [0] * (self.config['data']['text_length'] - length))
            text_turn.extend([self.vocab[f'#{turn}']] * length + [0] * (self.config['data']['text_length'] - length))
            text_speaker.extend([self.vocab[speaker]] * length + [0] * (self.config['data']['text_length'] - length))
        image_input = []
        image_pos = []
        image_turn = []
        image_speaker = []
        for turn, (img, img_length, speaker) in enumerate(zip(imgs, img_lengths, speakers)):
            image_input.extend(self.get_imgs(img))
            image_pos.extend([self.vocab[x] for x in self.image_pos[:img_length]] +
                             [0] * (self.config['data']['image_length'] - img_length))
            image_turn.extend([self.vocab[f'#{turn}']] * img_length +
                              [0] * (self.config['data']['image_length'] - img_length))
            image_speaker.extend([self.vocab[speaker]] * img_length +
                                 [0] * (self.config['data']['image_length'] - img_length))
        query_input, query_len = item[1]
        query_pos = list(range(1, query_len + 1)) + [0] * (self.config['data']['text_length'] - query_len)
        return np.array(text_input), np.array(text_pos), np.array(text_turn), np.array(text_speaker), \
               np.array(image_input, dtype=np.float32), np.array(image_pos), np.array(image_turn), np.array(
            image_speaker), np.array(query_input), np.array(query_pos)

    def __len__(self):
        return len(self.items)

    def get_img_urls(self, urls):
        ret = []
        for url in urls:
            if url in self.annoy_index:
                ret.append(url)
        return ret

    def get_imgs(self, urls):
        ret = []
        for url in urls:
            try:
                vector = self.annoy.get_item_vector(self.annoy_index[url])
            except:
                vector = [0.] * self.config['model']['image_size']
            ret.append(vector)
        return ret
