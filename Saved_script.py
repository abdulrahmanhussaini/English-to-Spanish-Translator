#!/usr/bin/env python3
# =============================================================================
# CIS433 Project 3 — English to Spanish Translator
# Standalone Inference Script
# Abdulrahman Hussaini | Simon Business School, University of Rochester
#
# Usage:
#   python inference.py
#
# Requirements:
#   pip install torch numpy
#
# The saved_model/ folder must be in the same directory as this script and
# must contain: best_model.pt, src_vocab.pkl, tgt_vocab.pkl, config.pkl
# rui_torch_transformer.py must also be in the same directory.
# =============================================================================

import re
import sys
import math
import pickle
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn


# =============================================================================
# 1. MultiHeadAttention cross-attention patch
#    Must be applied before the Transformer is instantiated.
#    The original rui_torch_transformer.py uses q.shape for k and v reshaping,
#    which crashes when src_len != tgt_len during cross-attention.
# =============================================================================

from rui_torch_transformer import Transformer, MultiHeadAttention


def _mha_forward(self, q, k, v):
    b, q_len, _ = q.shape
    kv_len = k.shape[1]

    q = self.W_q(q)
    k = self.W_k(k)
    v = self.W_v(v)

    q = q.view(b, q_len,  self.n_heads, self.d_k).transpose(1, 2)
    k = k.view(b, kv_len, self.n_heads, self.d_k).transpose(1, 2)
    v = v.view(b, kv_len, self.n_heads, self.d_k).transpose(1, 2)

    scores = q @ k.transpose(2, 3)
    if self.is_causal:
        mask = self.mask.bool()[:q_len, :q_len]
        scores.masked_fill_(mask, -torch.inf)

    weights = torch.softmax(scores / self.d_k ** 0.5, dim=-1)
    weights = self.dropout(weights)

    out = (weights @ v).transpose(1, 2)
    out = out.contiguous().view(b, q_len, self.d_model)
    return self.out_proj(out)


MultiHeadAttention.forward = _mha_forward


# =============================================================================
# 2. Vocab class
#    Must be defined before any pickle.load() call that references it.
#    Pickle cannot reconstruct the object without the class in memory.
# =============================================================================

class Vocab:
    PAD, UNK, SOS, EOS = 0, 1, 2, 3
    _SPECIALS = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}

    def __init__(self, max_size=14_000):
        self.max_size = max_size
        self.word2idx = dict(self._SPECIALS)
        self.idx2word = {v: k for k, v in self._SPECIALS.items()}

    def build(self, tokenized_sentences):
        freq = Counter(tok for sent in tokenized_sentences for tok in sent)
        for word, _ in freq.most_common(self.max_size - len(self._SPECIALS)):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def encode(self, tokens, add_sos=False, add_eos=False):
        ids = []
        if add_sos:
            ids.append(self.SOS)
        ids += [self.word2idx.get(t, self.UNK) for t in tokens]
        if add_eos:
            ids.append(self.EOS)
        return ids

    def decode(self, ids):
        out = []
        for i in ids:
            if i == self.EOS:
                break
            if i in (self.PAD, self.SOS):
                continue
            out.append(self.idx2word.get(i, '<UNK>'))
        return ' '.join(out)

    def __len__(self):
        return len(self.word2idx)


# =============================================================================
# 3. Tokenizer
# =============================================================================

def tokenize(text, lang='en'):
    text = text.lower().strip()
    text = re.sub(r'([?.!,\u00bf\u00a1;:])', r' \1 ', text)
    if lang == 'en':
        text = re.sub(r'[^a-z?.!,\u00bf\u00a1;: ]+', ' ', text)
    else:
        text = re.sub(r'[^a-z\u00e1\u00e9\u00ed\u00f3\u00fa\u00fc\u00f1?.!,\u00bf\u00a1;: ]+', ' ', text)
    return text.split()


# =============================================================================
# 4. Pipeline loader
# =============================================================================

def load_pipeline(model_dir, device):
    d = Path(model_dir)

    missing = [f for f in ['best_model.pt', 'src_vocab.pkl', 'tgt_vocab.pkl', 'config.pkl']
               if not (d / f).exists()]
    if missing:
        print(f'ERROR: Missing files in {d}: {missing}')
        sys.exit(1)

    with open(d / 'src_vocab.pkl', 'rb') as f:
        sv = pickle.load(f)
    with open(d / 'tgt_vocab.pkl', 'rb') as f:
        tv = pickle.load(f)
    with open(d / 'config.pkl', 'rb') as f:
        cfg = pickle.load(f)

    mdl = Transformer(
        n_layers       = cfg['n_layers'],
        d_emb          = cfg['d_emb'],
        n_heads        = cfg['n_heads'],
        d_ff           = cfg['d_ff'],
        src_vocab_size = cfg['src_vocab_size'],
        tgt_vocab_size = cfg['tgt_vocab_size'],
        seq_len        = cfg['max_len'] + 2,
        dropout        = cfg.get('dropout', 0.1),
    ).to(device)

    mdl.load_state_dict(torch.load(d / 'best_model.pt', map_location=device))
    mdl.eval()
    return mdl, sv, tv, cfg


# =============================================================================
# 5. Translate function
# =============================================================================

def translate(sentence, mdl, sv, tv, cfg, device, max_new=80):
    tokens  = tokenize(sentence, 'en')
    src_ids = sv.encode(tokens, add_eos=True)
    if not src_ids:
        return ''

    max_src = cfg['max_len']
    if len(src_ids) > max_src:
        src_ids = src_ids[:max_src - 1] + [sv.EOS]

    src_t   = torch.tensor([src_ids], dtype=torch.long).to(device)
    tgt_ids = [tv.SOS]
    max_tgt = cfg['max_len']

    with torch.no_grad():
        for _ in range(max_new):
            tgt_in  = tgt_ids[-max_tgt:]
            tgt_t   = torch.tensor([tgt_in], dtype=torch.long).to(device)
            logits  = mdl((src_t, tgt_t))
            logit_v = logits[0, -1, :].clone()

            for prev in set(tgt_ids):
                logit_v[prev] *= 0.7

            nxt = logit_v.argmax().item()
            tgt_ids.append(nxt)
            if nxt == tv.EOS:
                break

    return tv.decode(tgt_ids)


# =============================================================================
# 6. Main
# =============================================================================

def main():
    # Device: CPU only — no GPU assumed on the grader's machine
    device = torch.device('cpu')

    # Locate saved_model/ relative to this script's directory
    script_dir = Path(__file__).parent.resolve()
    model_dir  = script_dir / 'saved_model'

    print('=' * 60)
    print('  CIS433 Project 3 — English to Spanish Translator')
    print('  Loading model from:', model_dir)
    print('=' * 60)

    model, src_vocab, tgt_vocab, config = load_pipeline(model_dir, device)

    print('Model loaded successfully.')
    print(f"Architecture: d_emb={config['d_emb']}, n_layers={config['n_layers']}, "
          f"n_heads={config['n_heads']}, d_ff={config['d_ff']}")
    print(f"Vocab: {config['src_vocab_size']:,} EN tokens | {config['tgt_vocab_size']:,} ES tokens")
    print()
    print('Type an English sentence and press Enter to translate.')
    print('Type "quit" or "exit" to stop.')
    print('-' * 60)

    while True:
        try:
            text = input('English > ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nGoodbye!')
            break

        if not text:
            continue

        if text.lower() in ('quit', 'exit', 'q'):
            print('Goodbye!')
            break

        result = translate(text, model, src_vocab, tgt_vocab, config, device)
        print(f'Spanish > {result}')
        print()


if __name__ == '__main__':
    main()
