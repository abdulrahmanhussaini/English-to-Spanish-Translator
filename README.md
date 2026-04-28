# 🇪🇸 El Traductor — English to Spanish Neural Translator

A production-ready neural machine translation system built from scratch using a custom Transformer encoder-decoder architecture. Translates English text to Spanish with a BLEU score of **40.57** trained on 118,964 sentence pairs.

**Live demo → [english-to-spanish-translator-1.streamlit.app](https://english-to-spanish-translator-1.streamlit.app)**

---

## What This Demonstrates

- Building and training a **Transformer architecture from scratch** in PyTorch — no HuggingFace, no pretrained weights
- End-to-end NLP pipeline: raw text → custom tokenizer → vocabulary → encoder-decoder → greedy decode with repetition penalty
- Production deployment via **Streamlit Cloud** with a custom-themed UI
- Clean inference design: the grader/user loads saved weights and runs translation with zero retraining

---

## Architecture

| Parameter | Value |
|---|---|
| Model | Transformer Encoder-Decoder |
| Embedding dim | 128 |
| Layers | 2 encoder + 2 decoder |
| Attention heads | 8 |
| Feed-forward dim | 512 |
| Vocab (EN / ES) | 12,940 / 14,000 |
| Total parameters | ~6.2M |
| Training data | 118,964 pairs (full Tatoeba spa.txt) |
| Best val loss | 2.4810 |
| BLEU score | **40.57** |

---

## Stack

- **PyTorch** — model definition, training loop, inference
- **Streamlit** — web interface
- **Python** — custom tokenizer, Vocab class, greedy decoder, repetition penalty

---

## Run Locally

```bash
git clone https://github.com/abdulrahmanhussaini/English-to-Spanish-Translator
cd English-to-Spanish-Translator
pip install torch numpy streamlit
streamlit run app.py
```

Or run the command-line interface directly:

```bash
pip install torch numpy
python3 Saved_script.py
```

---

## Project Structure

```
├── app.py                   # Streamlit web app
├── Saved_script.py          # Standalone CLI inference script
├── rui_torch_transformer.py # Transformer architecture (encoder-decoder)
├── requirements.txt
└── saved_model/
    ├── best_model.pt        # Trained weights
    ├── src_vocab.pkl        # English vocabulary
    ├── tgt_vocab.pkl        # Spanish vocabulary
    └── config.pkl           # Model hyperparameters
```

---

## Sample Translations

| English | Spanish |
|---|---|
| I want to go to the library. | quiero ir a la biblioteca . |
| Let's party tonight. | esta noche vamos a la fiesta . |
| Hello, how are you? | hola , ¿ cómo estás ? |
| I do not understand. | no entiendo . |
| She is reading a book. | ella está leyendo un libro . |

---

*Built by [Abdulrahman Hussaini](https://github.com/abdulrahmanhussaini)*
