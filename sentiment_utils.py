"""
Shared sentiment-tagging utilities.

This is the single source of truth for how text gets tagged with
[NEGATIVE]/[NEUTRAL]/[POSITIVE] prefixes. bart_sentiment_controlled was
fine-tuned on 3-sentence-chunk tagging (see sentiment_tagging.ipynb) —
tag_text_with_sentiment_prefixes here MUST stay in sync with that, since it's
also used at inference time in evaluate.ipynb.
"""
import torch
import numpy as np
import nltk
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize

SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
LABELS = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
SENTENCES_PER_CHUNK = 3  # matches how the training data was tagged

device = "cuda" if torch.cuda.is_available() else "cpu"

_tokenizer = None
_model = None


def _load_sentiment_model():
    """Lazy-load so importing this module doesn't immediately pull the model."""
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME).to(device)
    return _tokenizer, _model


def get_sentiment_probs(text):
    """Return the [neg, neu, pos] probability vector for one chunk of text."""
    tokenizer, model = _load_sentiment_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return softmax(outputs.logits.cpu().numpy()[0])


def tag_text_with_sentiment_prefixes(text):
    """
    Groups text into chunks of SENTENCES_PER_CHUNK sentences and prefixes each
    chunk with its dominant sentiment. This is exactly how bart_sentiment_controlled's
    training data was built (add_sentiment_prefixes_by_nltk) — it must be the
    ONLY tagging function used, both for training data and at inference.
    """
    sentences = sent_tokenize(text)
    tagged_chunks = []
    current_chunk = []

    for sentence in sentences:
        if not sentence.strip():
            continue
        current_chunk.append(sentence.strip())
        if len(current_chunk) == SENTENCES_PER_CHUNK:
            chunk_text = " ".join(current_chunk)
            probs = get_sentiment_probs(chunk_text)
            tagged_chunks.append(f"[{LABELS[probs.argmax()]}] {chunk_text}")
            current_chunk = []

    if current_chunk:
        chunk_text = " ".join(current_chunk)
        probs = get_sentiment_probs(chunk_text)
        tagged_chunks.append(f"[{LABELS[probs.argmax()]}] {chunk_text}")

    return " ".join(tagged_chunks)


def add_sentiment_prefixes(example):
    """Dataset.map()-compatible wrapper. Adds an 'article_with_sentiment' field
    (matches the column name the training script tokenizes from)."""
    return {"article_with_sentiment": tag_text_with_sentiment_prefixes(example["article"])}


def get_text_sentiment_vector(text):
    """
    Average sentiment vector across paragraphs — used only for scoring alignment
    between a source text and a summary. Independent of the chunk-tagging above;
    no need for it to match the training format.
    """
    chunks = [c.strip() for c in text.split("\n") if c.strip()] or [text]
    vectors = [get_sentiment_probs(c) for c in chunks]
    return np.mean(vectors, axis=0)


def calculate_sentiment_preservation(source_text, summary_text):
    """
    Cosine similarity between source and summary sentiment vectors.
    1.0 = identical sentiment profile, -1.0 = opposite, 0 = unrelated.
    """
    v_source = get_text_sentiment_vector(source_text)
    v_summary = get_text_sentiment_vector(summary_text)
    score = float(np.dot(v_source, v_summary) / (np.linalg.norm(v_source) * np.linalg.norm(v_summary)))
    return {
        "source_vector_neg_neu_pos": v_source.tolist(),
        "summary_vector_neg_neu_pos": v_summary.tolist(),
        "sentiment_alignment_score": score,
    }
