import re
import sys
import pickle
import random
from pathlib import Path

import torch
import torch.nn as nn
import streamlit as st

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="El Traductor | English → Spanish",
    page_icon="🇪🇸",
    layout="centered",
)

# =============================================================================
# CULTURAL FACTS — shown randomly on each session
# =============================================================================
FACTS = [
    ("🎨", "La Sagrada Família", "Gaudí's basilica in Barcelona has been under construction since 1882 — and is still not finished."),
    ("💃", "El Flamenco", "Flamenco was declared an Intangible Cultural Heritage by UNESCO in 2010. It originated in Andalusia."),
    ("🍅", "La Tomatina", "Every August, the town of Buñol throws 150,000 tomatoes at each other in the world's largest food fight."),
    ("🐂", "San Fermín", "The Running of the Bulls in Pamplona lasts just 3 minutes — but participants train for months."),
    ("🌞", "La Siesta", "Spain has two peak hours of sunlight late in the day, which historically pushed lunch to 2–3 PM and dinner past 9 PM."),
    ("🏰", "La Alhambra", "The Alhambra palace in Granada contains over 10,000 individual geometric tile patterns — all unique."),
    ("🎭", "Don Quijote", "Cervantes' Don Quijote (1605) is considered the first modern novel and the best work of fiction ever written, per Nobel laureates."),
    ("🍷", "La Rioja", "Spain is the world's largest vineyard by surface area, with over 1 million hectares of wine-producing land."),
    ("⛪", "Santiago de Compostela", "The Camino de Santiago pilgrimage route attracts over 300,000 walkers per year from 180+ countries."),
    ("🎶", "El Idioma", "Spanish is the second most spoken native language in the world with over 490 million speakers — ahead of English."),
]

fact = random.choice(FACTS)

# =============================================================================
# INLINE CSS — full Spanish editorial theme
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;1,400&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Reset & base ──────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0f0a04 !important;
    color: #f5ede0 !important;
}

/* ── Streamlit frame cleanup ───────────────────────────────────────── */
.stApp { background-color: #0f0a04; }
section[data-testid="stSidebar"] { display: none; }
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 2rem !important;
    max-width: 720px !important;
}

/* ── Decorative top stripe ─────────────────────────────────────────── */
.stripe {
    height: 5px;
    background: linear-gradient(90deg, #c8102e 33%, #ffd700 33%, #ffd700 66%, #c8102e 66%);
    border-radius: 2px;
    margin-bottom: 2.2rem;
}

/* ── Hero header ───────────────────────────────────────────────────── */
.hero {
    text-align: center;
    margin-bottom: 2rem;
}
.hero-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: clamp(2.4rem, 6vw, 3.6rem);
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #f5ede0;
    line-height: 1.1;
    margin: 0 0 0.3rem 0;
}
.hero-title span {
    color: #c8102e;
    font-style: italic;
}
.hero-sub {
    font-size: 0.9rem;
    color: #a0896e;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    font-weight: 300;
}

/* ── Fact card ─────────────────────────────────────────────────────── */
.fact-card {
    background: linear-gradient(135deg, #1c120a 0%, #251a0e 100%);
    border: 1px solid #3a2710;
    border-left: 4px solid #c8102e;
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 1.8rem;
    display: flex;
    gap: 1rem;
    align-items: flex-start;
}
.fact-icon { font-size: 1.8rem; flex-shrink: 0; margin-top: 2px; }
.fact-label {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #c8102e;
    margin-bottom: 0.2rem;
}
.fact-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #f5ede0;
    margin-bottom: 0.2rem;
}
.fact-body {
    font-size: 0.85rem;
    color: #a0896e;
    line-height: 1.5;
}

/* ── Section label ─────────────────────────────────────────────────── */
.section-label {
    font-size: 0.72rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #7a6248;
    font-weight: 500;
    margin-bottom: 0.5rem;
}

/* ── Textarea override ─────────────────────────────────────────────── */
textarea {
    background-color: #1c120a !important;
    color: #f5ede0 !important;
    border: 1px solid #3a2710 !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1.05rem !important;
    caret-color: #c8102e !important;
    transition: border-color 0.2s !important;
}
textarea:focus {
    border-color: #c8102e !important;
    box-shadow: 0 0 0 2px rgba(200, 16, 46, 0.18) !important;
}
textarea::placeholder { color: #4a3520 !important; }

/* ── Button ────────────────────────────────────────────────────────── */
.stButton > button {
    background: #c8102e !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    padding: 0.65rem 2rem !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: background 0.2s, transform 0.12s !important;
}
.stButton > button:hover {
    background: #a50d26 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Result box ────────────────────────────────────────────────────── */
.result-box {
    background: linear-gradient(135deg, #1c120a 0%, #251a0e 100%);
    border: 1px solid #3a2710;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-top: 1.4rem;
    position: relative;
    overflow: hidden;
}
.result-box::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #c8102e, #ffd700, #c8102e);
}
.result-flag {
    font-size: 0.75rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #7a6248;
    margin-bottom: 0.5rem;
}
.result-text {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.55rem;
    font-style: italic;
    color: #f5ede0;
    line-height: 1.4;
    word-break: break-word;
}

/* ── Error box ─────────────────────────────────────────────────────── */
.error-box {
    background: #1c0a0a;
    border: 1px solid #5a1010;
    border-radius: 10px;
    padding: 1rem 1.3rem;
    margin-top: 1.2rem;
    color: #e07070;
    font-size: 0.9rem;
}

/* ── Divider ───────────────────────────────────────────────────────── */
.divider {
    border: none;
    border-top: 1px solid #2a1e12;
    margin: 2rem 0;
}

/* ── Footer ────────────────────────────────────────────────────────── */
.footer {
    text-align: center;
    font-size: 0.75rem;
    color: #4a3520;
    margin-top: 2.5rem;
    letter-spacing: 0.1em;
}
.footer span { color: #c8102e; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# MHA PATCH — must happen before load_pipeline instantiates the model
# =============================================================================

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


from rui_torch_transformer import Transformer, MultiHeadAttention
MultiHeadAttention.forward = _mha_forward


# =============================================================================
# VOCAB + TOKENIZER
# =============================================================================

from collections import Counter

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
        if add_sos: ids.append(self.SOS)
        ids += [self.word2idx.get(t, self.UNK) for t in tokens]
        if add_eos: ids.append(self.EOS)
        return ids

    def decode(self, ids):
        out = []
        for i in ids:
            if i == self.EOS: break
            if i in (self.PAD, self.SOS): continue
            out.append(self.idx2word.get(i, '<UNK>'))
        return ' '.join(out)

    def __len__(self):
        return len(self.word2idx)


def tokenize(text, lang='en'):
    text = text.lower().strip()
    text = re.sub(r'([?.!,\u00bf\u00a1;:])', r' \1 ', text)
    if lang == 'en':
        text = re.sub(r'[^a-z?.!,\u00bf\u00a1;: ]+', ' ', text)
    else:
        text = re.sub(r'[^a-z\u00e1\u00e9\u00ed\u00f3\u00fa\u00fc\u00f1?.!,\u00bf\u00a1;: ]+', ' ', text)
    return text.split()


# =============================================================================
# PIPELINE LOADER — cached so it only loads once per session
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_pipeline():
    device = torch.device('cpu')
    d = Path(__file__).parent

    with open(d / 'src_vocab.pkl', 'rb') as f: sv  = pickle.load(f)
    with open(d / 'tgt_vocab.pkl', 'rb') as f: tv  = pickle.load(f)
    with open(d / 'config.pkl',    'rb') as f: cfg = pickle.load(f)

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
    return mdl, sv, tv, cfg, device


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
# UI
# =============================================================================

# Spanish flag stripe
st.markdown('<div class="stripe"></div>', unsafe_allow_html=True)

# Hero
st.markdown("""
<div class="hero">
    <div class="hero-title">El <span>Traductor</span></div>
    <div class="hero-sub">English &nbsp;→&nbsp; Spanish &nbsp;·&nbsp; Transformer-Powered</div>
</div>
""", unsafe_allow_html=True)

# Cultural fact card
st.markdown(f"""
<div class="fact-card">
    <div class="fact-icon">{fact[0]}</div>
    <div>
        <div class="fact-label">¿Sabías que? &nbsp;·&nbsp; Did you know?</div>
        <div class="fact-title">{fact[1]}</div>
        <div class="fact-body">{fact[2]}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Load model
try:
    model, src_vocab, tgt_vocab, config, device = load_pipeline()
    model_ok = True
except Exception as e:
    model_ok = False
    st.markdown(f'<div class="error-box">⚠️ Could not load model: {e}</div>',
                unsafe_allow_html=True)

# Input
st.markdown('<div class="section-label">🇬🇧 &nbsp; Enter English text</div>', unsafe_allow_html=True)
user_input = st.text_area(
    label="English input",
    placeholder="e.g.  I want to visit Barcelona next summer.",
    height=110,
    label_visibility="collapsed",
)

translate_btn = st.button("Traducir →", disabled=not model_ok)

# Output
if translate_btn:
    text = user_input.strip()
    if not text:
        st.markdown('<div class="error-box">Please enter an English sentence above.</div>',
                    unsafe_allow_html=True)
    else:
        with st.spinner("Translating…"):
            result = translate(text, model, src_vocab, tgt_vocab, config, device)

        if result:
            st.markdown(f"""
<div class="result-box">
    <div class="result-flag">🇪🇸 &nbsp; Spanish translation</div>
    <div class="result-text">{result}</div>
</div>
""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-box">Translation returned empty. Try a different sentence.</div>',
                        unsafe_allow_html=True)

# Divider + example phrases
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Try these phrases</div>', unsafe_allow_html=True)

examples = [
    "Hello, how are you?",
    "I want to go to the library.",
    "The weather is beautiful today.",
    "I do not understand.",
    "She is reading a book.",
]

cols = st.columns(len(examples))
for col, phrase in zip(cols, examples):
    with col:
        if st.button(phrase, key=phrase):
            with st.spinner("Translating…"):
                result = translate(phrase, model, src_vocab, tgt_vocab, config, device)
            st.markdown(f"""
<div class="result-box">
    <div class="result-flag">🇪🇸 &nbsp; Spanish translation</div>
    <div class="result-text">{result}</div>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    Built with <span>♥</span> for CIS433 &nbsp;·&nbsp; Simon Business School &nbsp;·&nbsp;
    Transformer: d_emb=128, n_layers=2, n_heads=8, d_ff=512
</div>
""", unsafe_allow_html=True)
