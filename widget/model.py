# -*- coding: utf-8 -*-  
"""
@Author: winton
@File: widget.py
@Time: 2019/8/6 9:13 AM
@Description:
"""
import pickle

import torch
from torch import nn

from .module import Emb, Encoder, Decoder


def get_non_pad_mask(seq, padding_id):
    assert seq.dim() == 2
    return seq.ne(padding_id).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q, padding_id):
    """ For masking out the padding part of key sequence. """

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(padding_id)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


def get_subsequent_mask(seq):
    """ For masking out the subsequent info. """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask


class Model(nn.Module):
    def __init__(self,
                 task,
                 vocab_size,
                 max_text_len,
                 image_size,
                 embedding_size,
                 text_n_layers, text_n_head, text_d_k, text_d_v, text_d_model, text_d_inner,
                 co_n_layers, co_n_head, co_d_k, co_d_v, co_d_model, co_d_inner,
                 de_n_layers, de_n_head, de_d_k, de_d_v, de_d_model, de_d_inner,
                 dropout_rate,
                 padding_id,
                 tgt_emb_prj_weight_sharing=True):
        super().__init__()
        self.task = task
        self.vocab_size = vocab_size
        self.padding_id = padding_id
        self.emb = Emb(max_text_len, image_size, vocab_size, embedding_size, padding_id)
        self.text_encoder = Encoder(text_n_layers, text_n_head, text_d_k, text_d_v,
                                    text_d_model, text_d_inner, dropout_rate)
        self.text_co_encoder = Encoder(co_n_layers, co_n_head, co_d_k, co_d_v,
                                       co_d_model, co_d_inner, dropout_rate)
        self.image_co_encoder = Encoder(co_n_layers, co_n_head, co_d_k, co_d_v,
                                        co_d_model, co_d_inner, dropout_rate)
        if self.task == 'text':
            self.decoder = Decoder(de_n_layers, de_n_head, de_d_k, de_d_v,
                                   de_d_model, de_d_inner, dropout_rate)
            self.tgt_word_prj = nn.Linear(de_d_model, vocab_size, bias=False)
            if tgt_emb_prj_weight_sharing:
                # Share the weight matrix between target word embedding & the final logit dense layer
                self.tgt_word_prj.weight = self.emb.tgt_token_emb.weight
                self.x_logit_scale = (de_d_model ** -0.5)
            else:
                nn.init.xavier_normal_(self.tgt_word_prj.weight)
                self.x_logit_scale = 1.
        # else:
        #     self.image_similarity_encoder = ImageEncoder(self.image_in_size, self.context_hidden_size)

    def forward(self, context, query):
        if self.task == 'text':
            context_embs, context_seq = self.context_encode(context)
            return self.text_decode(query, context_embs, context_seq)

    def context_encode(self, context):
        text_input, text_pos, text_turn, text_speaker, image_input, image_pos, image_turn, image_speaker = context
        text_embs, image_embs = self.emb(text_input, text_pos, text_turn, text_speaker,
                                         image_input, image_pos, image_turn, image_speaker)
        # text_embs = (bs, text_len, embedding_size)
        # image_embs = (bs, image_len, embedding_size)
        # -- text transformer encoder
        attn_mask = get_attn_key_pad_mask(seq_k=text_input, seq_q=text_input, padding_id=self.padding_id)
        non_pad_mask = get_non_pad_mask(text_input, self.padding_id)
        text_embs, = self.text_encoder(text_embs, text_embs, non_pad_mask, attn_mask)  # (bs, text_len, embedding_size)
        # -- co-attention encoder
        attn_mask = get_attn_key_pad_mask(seq_k=image_pos, seq_q=text_input, padding_id=self.padding_id)
        non_pad_mask = get_non_pad_mask(text_input, self.padding_id)
        text_co_embs, = self.text_co_encoder(image_embs, text_embs, non_pad_mask, attn_mask)
        # text_co_embs = (bs, text_len, embedding_size)
        attn_mask = get_attn_key_pad_mask(seq_k=text_input, seq_q=image_pos, padding_id=self.padding_id)
        non_pad_mask = get_non_pad_mask(image_pos, self.padding_id)
        image_co_embs, = self.image_co_encoder(text_embs, image_embs, non_pad_mask, attn_mask)
        # image_co_embs = (bs, image_len, embedding_size)
        context_embs = torch.cat((text_co_embs, image_co_embs), dim=1)
        # context_embs = (bs, text_len + image_len, embedding_size)
        context_seq = torch.cat((text_input, image_pos), dim=1)
        return context_embs, context_seq

    def text_decode(self, query, context_embs, context_seq):
        query_input, query_pos = query
        query_embs = self.emb.tgt_token_emb(query_input) + self.emb.position_enc(query_pos)
        # query_embs = (bs, text_len, embedding_size)
        # -- query transformer decoder
        non_pad_mask = get_non_pad_mask(query_input, self.padding_id)
        slf_attn_mask_subseq = get_subsequent_mask(query_input)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=query_input, seq_q=query_input,
                                                     padding_id=self.padding_id)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=context_seq, seq_q=query_input, padding_id=self.padding_id)
        dec_output, = self.decoder(context_embs, query_embs, non_pad_mask, slf_attn_mask, dec_enc_attn_mask)
        # dec_output = (bs, query_len, embedding_size)
        seq_logit = self.tgt_word_prj(dec_output) * self.x_logit_scale
        # seq_logit = (bs, query_len, vocab_size)
        return seq_logit.view(-1, seq_logit.size(2))


if __name__ == '__main__':
    pass
    # vocab = pickle.load(open('dataset/transformer/v2/c2/vocab.pkl', 'rb'))
    # widget = Model(task='text',
    #               vocab_size=len(vocab),
    #               max_text_len=20,
    #               image_size=4096,
    #               embedding_size=512,
    #               text_n_layers=6, text_n_head=8, text_d_k=64, text_d_v=64, text_d_model=512, text_d_inner=2048,
    #               co_n_layers=6, co_n_head=8, co_d_k=64, co_d_v=64, co_d_model=512, co_d_inner=2048,
    #               de_n_layers=6, de_n_head=8, de_d_k=64, de_d_v=64, de_d_model=512, de_d_inner=2048,
    #               dropout_rate=0.1,
    #               padding_id=0)
    # # context_size = 2
    # text_input = torch.tensor([[1, 2, 3, 4, 5, 0, 0, 1, 2, 3, 0, 0, 0, 0],
    #                            [1, 2, 3, 4, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0]])
    # text_pos = torch.tensor([[1, 2, 3, 4, 5, 0, 0, 1, 2, 3, 0, 0, 0, 0],
    #                          [1, 2, 3, 4, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0]])
    # text_turn = torch.tensor([[6, 6, 6, 6, 6, 0, 0, 7, 7, 7, 0, 0, 0, 0],
    #                           [6, 6, 6, 6, 0, 0, 0, 7, 7, 0, 0, 0, 0, 0]])
    # text_speaker = torch.tensor([[8, 8, 8, 8, 8, 0, 0, 9, 9, 9, 0, 0, 0, 0],
    #                              [8, 8, 8, 8, 0, 0, 0, 9, 9, 0, 0, 0, 0, 0]])
    # image_input = torch.rand(size=(2, 10, 4096))
    # image_seq = torch.tensor([[1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
    #                           [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]])
    # image_turn = torch.tensor([[6, 6, 0, 0, 0, 7, 7, 7, 0, 0],
    #                            [6, 6, 6, 6, 6, 7, 0, 0, 0, 0]])
    # image_speaker = torch.tensor([[8, 8, 0, 0, 0, 9, 9, 9, 0, 0],
    #                               [8, 8, 8, 8, 8, 9, 0, 0, 0, 0]])
    # query_input = torch.tensor([[1, 2, 3, 4, 0, 0],
    #                             [1, 5, 2, 0, 0, 0]])
    # query_pos = torch.tensor([[1, 2, 3, 4, 0, 0],
    #                           [1, 2, 3, 0, 0, 0]])
    # widget(text_input, text_pos, text_turn, text_speaker,
    #       image_input, image_seq, image_turn, image_speaker,
    #       query_input, query_pos)
