# -*- coding: utf-8 -*-  
"""
@Author: winton
@File: test.py
@Time: 2019/8/8 6:54 PM
@Description:
"""
import argparse
import json
import logging
import os
import pickle

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from widget.baseline.transformer.data import DataSource
from widget.baseline.transformer.model import Model


def index2word(seq, vocab, end_id):
    words = []
    for word_id in seq:
        if word_id == end_id:
            break
        word = vocab[word_id]
        words.append(word)
    return ' '.join(words)


def main(args):
    torch.set_default_tensor_type(torch.FloatTensor)
    # Load config
    config = json.load(open(args.config_file_path, 'r'))
    # load save dict
    save_dict = torch.load(os.path.join(args.model_path, args.checkpoint_file), map_location=f'cuda:0')
    task = save_dict['task']
    # Set logger (console and file)
    logger_format = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    logger = logging.getLogger('saha')
    sh = logging.StreamHandler()
    sh.setFormatter(logger_format)
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)
    fh = logging.FileHandler(os.path.join(args.model_path, f'test_{task}.log'), 'a', encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logger_format)
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)
    logger.info(json.dumps(config, indent=2))
    # Set device and seed
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    # load data
    logger.info('reading vocab pkl...')
    vocab = pickle.load(open(config['data']['vocab_file'], 'rb'))
    test_dataset = DataSource(args.config_file_path, vocab, task, 'test')
    test_loader = DataLoader(test_dataset,
                             batch_size=config['training']['valid_batch_size'],
                             shuffle=False,
                             num_workers=4)
    vocab_size = len(vocab)
    i2w = {v: k for (k, v) in vocab.items()}
    # Define widget
    model = Model(task=task,
                  vocab_size=vocab_size,
                  max_text_len=config['data']['text_length'],
                  image_size=config['widget']['image_size'],
                  embedding_size=config['widget']['word_embedding_size'],
                  text_n_layers=config['widget']['text_n_layers'],
                  text_n_head=config['widget']['text_n_head'],
                  text_d_k=config['widget']['text_d_k'],
                  text_d_v=config['widget']['text_d_v'],
                  text_d_model=config['widget']['text_d_model'],
                  text_d_inner=config['widget']['text_d_inner'],
                  co_n_layers=config['widget']['co_n_layers'],
                  co_n_head=config['widget']['co_n_head'],
                  co_d_k=config['widget']['co_d_k'],
                  co_d_v=config['widget']['co_d_v'],
                  co_d_model=config['widget']['co_d_model'],
                  co_d_inner=config['widget']['co_d_inner'],
                  de_n_layers=config['widget']['de_n_layers'],
                  de_n_head=config['widget']['de_n_head'],
                  de_d_k=config['widget']['de_d_k'],
                  de_d_v=config['widget']['de_d_v'],
                  de_d_model=config['widget']['de_d_model'],
                  de_d_inner=config['widget']['de_d_inner'],
                  dropout_rate=config['widget']['dropout_rate'],
                  padding_id=config['data']['pad_id'],
                  tgt_emb_prj_weight_sharing=True)
    model.load_state_dict(save_dict['widget'])
    # widget = nn.DataParallel(widget)
    model.to(device)
    model.eval()
    logger.info(model)
    true_sequences = []
    pred_sequences = []
    prog = tqdm(total=len(test_dataset) // config['training']['valid_batch_size'])
    with torch.no_grad():
        if task == 'text':
            for batch_data in test_loader:
                text_input, text_pos, text_turn, text_speaker, \
                image_input, image_seq, image_turn, image_speaker, \
                query_input, query_pos = map(lambda x: x.to(device), batch_data)
                for sequence in query_input.cpu().numpy():
                    true_sequences.append(index2word(sequence, i2w, config['data']['end_id']))
                context_embs, context_seq = model.context_encode((text_input, text_pos, text_turn, text_speaker,
                                                                 image_input, image_seq, image_turn, image_speaker))
                pred_text = query_input[:, :1]
                for len_dec_seq in range(1, config['data']['text_length'] + 1):
                    dec_output_prob = model.text_decode((pred_text, query_pos[:, :len_dec_seq]), context_embs, context_seq)
                    dec_output_prob = dec_output_prob.view(-1, len_dec_seq, vocab_size)
                    _, max_text = torch.max(torch.softmax(dec_output_prob, dim=2), dim=2)
                    current_text = max_text[:, -1].view(-1, 1)
                    pred_text = torch.cat((pred_text, current_text), dim=1)
                for sequence in pred_text.cpu().numpy():
                    pred_sequences.append(index2word(sequence, i2w, config['data']['end_id']))
                prog.update()
            prog.close()
            with open(os.path.join(args.model_path, args.out_file), 'w') as f:
                for item in pred_sequences:
                    f.write(f"{item}\n")
            if not os.path.isfile(os.path.join(args.model_path, 'gt_text.txt')):
                with open(os.path.join(args.model_path, 'gt_text.txt'), 'w') as f:
                    for item in true_sequences:
                        f.write(f"{item}\n")


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    # cuda device
    _parser.add_argument('-g', '--gpu', default='0', help='choose which GPU to use')
    # path
    _parser.add_argument('--config_file_path', help='path to json config', required=True)
    _parser.add_argument('--model_path', type=str, default='./models/', help='path for trained models')
    _parser.add_argument('--checkpoint_file', help='checkpoint file', required=True)
    _parser.add_argument('--out_file', type=str, help='path for saving result', required=True)
    _args = _parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = _args.gpu
    exit(main(_args))
