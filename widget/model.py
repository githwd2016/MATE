# -*- coding: utf-8 -*-  
"""
@Author: winton
@File: widget.py
@Time: 2019/8/6 9:13 AM
@Description:
"""
import torch
from torch import nn

from .module import Emb, Encoder, Decoder, KnowledgeDecoder


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
                 tgt_emb_prj_weight_sharing=True,
                 use_knowledge=False,
                 knowledge_data=None):
        super().__init__()
        self.task = task
        self.vocab_size = vocab_size
        self.padding_id = padding_id
        self.use_knowledge = use_knowledge
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
            self.tgt_word_prj_1 = nn.Linear(de_d_model, vocab_size, bias=False)
            if tgt_emb_prj_weight_sharing:
                # Share the weight matrix between target word embedding & the final logit dense layer
                self.tgt_word_prj_1.weight = self.emb.tgt_token_emb.weight
                self.x_logit_scale = (de_d_model ** -0.5)
            else:
                nn.init.xavier_normal_(self.tgt_word_prj_1.weight)
                self.x_logit_scale = 1.
            if use_knowledge:
                self.knowledge_data = knowledge_data
                self.knowledge_linear = nn.Linear(2 * embedding_size, embedding_size)
                self.knowledge_decoder = KnowledgeDecoder(de_n_layers, de_n_head, de_d_k, de_d_v,
                                                          de_d_model, de_d_inner, dropout_rate)
                self.tgt_word_prj_2 = nn.Linear(de_d_model, vocab_size, bias=False)
                if tgt_emb_prj_weight_sharing:
                    # Share the weight matrix between target word embedding & the final logit dense layer
                    self.tgt_word_prj_2.weight = self.emb.tgt_token_emb.weight
                else:
                    nn.init.xavier_normal_(self.tgt_word_prj_2.weight)

    def forward(self, context, query):
        if self.task == 'text':
            context_embs, context_seq = self.context_encode(context)
            output_1 = self.text_decode(query, context_embs, context_seq)
            seq_logit_1 = self.tgt_word_prj_1(output_1) * self.x_logit_scale
            seq_logit_1 = seq_logit_1.view(-1, seq_logit_1.size(2))
            if self.use_knowledge:
                knowledge = self.emb.tgt_token_emb(self.knowledge_data)
                knowledge = self.knowledge_linear(torch.reshape(knowledge, [-1, 2 * knowledge.shape[2]]))
                knowledge = knowledge.unsqueeze(0).expand(context_embs.shape[0], knowledge.shape[0], knowledge.shape[1])
                output_2 = self.knowledge_text_decode(query, context_embs, context_seq, knowledge)
                seq_logit_2 = self.tgt_word_prj_2(output_2) * self.x_logit_scale
                seq_logit_2 = seq_logit_2.view(-1, seq_logit_2.size(2))
                return seq_logit_1, seq_logit_2
            else:
                return seq_logit_1

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
        # seq_logit = self.tgt_word_prj_1(dec_output) * self.x_logit_scale
        # seq_logit = (bs, query_len, vocab_size)
        return dec_output

    def knowledge_text_decode(self, query, context_embs, context_seq, knowledge):
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
        dec_output, = self.knowledge_decoder(context_embs, query_embs, knowledge,
                                             non_pad_mask, slf_attn_mask, dec_enc_attn_mask)
        # dec_output = (bs, query_len, embedding_size)
        # seq_logit = self.tgt_word_prj_2(dec_output) * self.x_logit_scale
        # seq_logit = (bs, query_len, vocab_size)
        return dec_output


if __name__ == '__main__':
    pass
