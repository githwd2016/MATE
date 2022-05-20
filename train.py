# -*- coding: utf-8 -*-  
"""
@Author: winton
@File: training.py
@Time: 2021/4/9 9:22 AM
@Description:
"""
import argparse
import json
import logging
import os

import torch
from torch import nn, optim
from torch.nn.functional import cross_entropy
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import numpy as np

from widget.data import DataSource, PAD_ID
from widget.model import Model


def cal_performance(pred, gold, padding_id, smoothing=False):
    """ Apply label smoothing if needed """

    loss = cal_loss(pred, gold, smoothing, padding_id)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(padding_id)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct


def cal_loss(pred, gold, smoothing, padding_id):
    """ Calculate cross entropy loss, apply label smoothing if needed. """

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = torch.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(padding_id)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = cross_entropy(pred, gold, ignore_index=padding_id, reduction='sum')

    return loss


def main(args):
    torch.set_default_tensor_type(torch.FloatTensor)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    # Load config
    config = json.load(open(args.config_file_path, 'r'))
    if config['training']['label_smoothing'] == 1:
        label_smoothing = True
    else:
        label_smoothing = False
    # Set logger (console and file)
    logger_format = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    logger = logging.getLogger('transformer')
    sh = logging.StreamHandler()
    sh.setFormatter(logger_format)
    sh.setLevel(logging.DEBUG)
    logger.addHandler(sh)
    fh = logging.FileHandler(os.path.join(args.model_path, f'training_{args.task}.log'), 'a', encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logger_format)
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)
    logger.info(json.dumps(config, indent=2))
    # Set device and seed
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(config['training']['seed'])  # Seed for reproducing
    # load data
    train_dataset = DataSource(args.config_file_path, args.task, 'train',
                               args.version, args.context_size)
    valid_dataset = DataSource(args.config_file_path, args.task, 'valid',
                               args.version, args.context_size)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=3)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=3)
    knowledge_data = train_dataset.encode_knowledge_pair(config['data']['knowledge_path']).to(device)
    vocab_size = len(train_dataset.vocab)
    logger.info(f"Total batches={len(train_dataset) // args.batch_size}")
    # Define widget
    model = Model(task=args.task,
                  vocab_size=vocab_size,
                  max_text_len=config['data']['text_length'],
                  image_size=config['model']['image_size'],
                  embedding_size=config['model']['word_embedding_size'],
                  text_n_layers=config['model']['text_n_layers'],
                  text_n_head=config['model']['text_n_head'],
                  text_d_k=config['model']['text_d_k'],
                  text_d_v=config['model']['text_d_v'],
                  text_d_model=config['model']['text_d_model'],
                  text_d_inner=config['model']['text_d_inner'],
                  co_n_layers=config['model']['co_n_layers'],
                  co_n_head=config['model']['co_n_head'],
                  co_d_k=config['model']['co_d_k'],
                  co_d_v=config['model']['co_d_v'],
                  co_d_model=config['model']['co_d_model'],
                  co_d_inner=config['model']['co_d_inner'],
                  de_n_layers=config['model']['de_n_layers'],
                  de_n_head=config['model']['de_n_head'],
                  de_d_k=config['model']['de_d_k'],
                  de_d_v=config['model']['de_d_v'],
                  de_d_model=config['model']['de_d_model'],
                  de_d_inner=config['model']['de_d_inner'],
                  dropout_rate=config['model']['dropout_rate'],
                  padding_id=PAD_ID,
                  tgt_emb_prj_weight_sharing=True,
                  use_knowledge=config['model']['use_knowledge'],
                  knowledge_data=knowledge_data
                  )
    model.to(device)
    # model = nn.DataParallel(model)
    # logger.info(model)
    # Define optimizer
    optimizer = optim.Adam(model.parameters(),
                           lr=config['training']['lr'],
                           weight_decay=config['training']['lr_decay'])
    # Define learning rate scheduler
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
    #                                            factor=0.5, patience=1,
    #                                            threshold=0.1, threshold_mode='rel',
    #                                            cooldown=0, min_lr=1e-8,
    #                                            verbose=True)
    # optimizer = ScheduledOptim(
    #     optim.Adam(
    #         filter(lambda x: x.requires_grad, widget.parameters()),
    #         betas=(0.9, 0.98), eps=1e-09),
    #     config['widget']['text_d_model'], config['training']['warmup_steps'])
    # Train
    total_batch = 0
    min_val_loss = None
    bad_loss_cnt = 0
    for epoch in range(config['training']['num_epochs']):
        total_losses = [0, 0]
        n_word_total_list = [0, 0]
        n_word_correct_list = [0, 0]
        # scheduler.step(sum(valid_losses))
        for train_batch in train_loader:
            total_batch += 1
            if args.task == 'text':
                text_input, text_pos, text_turn, text_speaker, \
                image_input, image_pos, image_turn, image_speaker, \
                query_input, query_pos = map(lambda x: x.to(device), train_batch)
                gold = query_input[:, 1:]
                optimizer.zero_grad()
                dec_output_probs = model((text_input, text_pos, text_turn, text_speaker,
                                          image_input, image_pos, image_turn, image_speaker),
                                         (query_input[:, :-1], query_pos[:, :-1]))
                backward_loss = []
                for idx, dec_output_prob in enumerate(dec_output_probs):
                    loss, n_correct = cal_performance(dec_output_prob, gold, PAD_ID,
                                                      smoothing=label_smoothing)
                    backward_loss.append(loss)
                    batch_loss = loss.item()
                    total_losses[idx] += batch_loss
                    non_pad_mask = gold.ne(PAD_ID)
                    n_word = non_pad_mask.sum().item()
                    n_word_total_list[idx] += n_word
                    n_word_correct_list[idx] += n_correct
                    if total_batch % config['training']['log_batch'] == 0 or total_batch < config['training'][
                        'log_batch']:
                        logger.info(f'Epoch [{epoch + 1}], Batch [{total_batch}], '
                                    f'Loss {idx + 1}: {batch_loss / n_word:.6}, '
                                    f'Accuracy {idx + 1}: {100 * n_correct / n_word:.3f} %')
                backward_loss = sum(backward_loss)
                backward_loss.backward()
                optimizer.step()
                # optimizer.step_and_update_lr()
                # Gradient clipping to avoid exploding gradients
                nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_gradient_norm'])
            else:
                pass
        if args.task == 'text':
            for idx, (total_loss, n_word_total, n_word_correct) in enumerate(
                    zip(total_losses, n_word_total_list, n_word_correct_list)):
                train_loss = total_loss / n_word_total
                train_accu = n_word_correct / n_word_total
                logger.info(f'Epoch [{epoch + 1}], '
                            f'Train Loss {idx + 1}: {train_loss:.6}, '
                            f'Train Accuracy {idx + 1}: {100 * train_accu:.3f} %')
        # Evaluate
        valid_losses = [0, 0]
        n_word_total_list = [0, 0]
        n_word_correct_list = [0, 0]
        if epoch % config['training']['evaluate_epoch'] == 0:
            model.eval()
            if args.task == 'text':
                for valid_batch in valid_loader:
                    text_input, text_pos, text_turn, text_speaker, \
                    image_input, image_pos, image_turn, image_speaker, \
                    query_input, query_pos = map(lambda x: x.to(device), valid_batch)
                    gold = query_input[:, 1:]
                    dec_output_probs = model((text_input, text_pos, text_turn, text_speaker,
                                              image_input, image_pos, image_turn, image_speaker),
                                             (query_input[:, :-1], query_pos[:, :-1]))
                    for idx, dec_output_prob in enumerate(dec_output_probs):
                        loss_val, n_correct = cal_performance(dec_output_prob, gold, PAD_ID,
                                                              smoothing=label_smoothing)
                        valid_losses[idx] += loss_val.item()
                        non_pad_mask = gold.ne(PAD_ID)
                        n_word = non_pad_mask.sum().item()
                        n_word_total_list[idx] += n_word
                        n_word_correct_list[idx] += n_correct
                for idx, (valid_loss, n_word_total, n_word_correct) in enumerate(
                        zip(valid_losses, n_word_total_list, n_word_correct_list)):
                    logger.info(f'Epoch [{epoch + 1}] '
                                f'Valid Loss {idx + 1}: {valid_loss / n_word_total:.6}, '
                                f'Valid Accuracy {idx + 1}: {100 * n_word_correct / n_word_total:.3f} %, '
                                f'Patience: {bad_loss_cnt}')
                valid_loss = sum(valid_losses)
                model.train()
                # Save widget each epoch
                save_dict = {
                    'task': args.task,
                    'epoch': epoch,
                    'iteration': total_batch,
                    'valid_loss': valid_loss,
                    'widget': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                torch.save(save_dict,
                           os.path.join(args.model_path, f'{args.task}_model_{epoch + 1}.pth'))
                if min_val_loss is None or valid_loss < min_val_loss:
                    min_val_loss = valid_loss
                    bad_loss_cnt = 0
                    # Save the best widget
                    torch.save(save_dict,
                               os.path.join(args.model_path, f'best_{args.task}_model.pth'))
                else:
                    bad_loss_cnt += 1
                    if bad_loss_cnt >= config['training']['patience']:
                        return 0
            else:
                pass
    return 0


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    # cuda device
    _parser.add_argument('-g', '--gpu', default='0', help='choose which GPU to use')
    # path
    _parser.add_argument('--config_file_path', help='path to json config', required=True)
    _parser.add_argument('--model_path', type=str, default='./models/', help='path for saving trained models')
    # widget
    _parser.add_argument('--task', type=str, default='text', help='task type(only support text now).')
    _parser.add_argument('--version', type=int, choices=[1, 2], help='dataset version.', required=True)
    _parser.add_argument('--context_size', type=int, help='context size.', required=True)
    _parser.add_argument('--batch_size', type=int, help='batch size.', required=True)
    _args = _parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = _args.gpu
    exit(main(_args))
