# Sentiment-Aware Abstractive Summarization

Fine-tuning BART for news summarization, then extending it with a sentiment-control
mechanism: articles are tagged with their emotional arc before summarization, so the
model can be steered to preserve tone, not just content.

## Results

**SAMSum (dialogue summarization), fine-tuned vs. base:**

| Metric    | Fine-tuned | Base  |
|-----------|-----------|-------|
| ROUGE-1   | 0.498     | 0.332 |
| ROUGE-2   | 0.290     | 0.112 |
| ROUGE-L   | 0.425     | 0.260 |
| ROUGE-Lsum| 0.427     | 0.261 |

Fine-tuning clearly helped here — SAMSum's dialogue format is different enough from
BART's original CNN/DailyMail pretraining that adapting to it paid off.

**CNN/DailyMail, plain fine-tune vs. base:**

Fine-tuning BART directly on CNN/DailyMail *hurt* ROUGE relative to the untouched
base model (`facebook/bart-large-cnn` is already pretrained on this exact dataset,
so further fine-tuning on a small slice mostly just overfits). The base model was
kept for CNN summarization; the fine-tuned checkpoint was discarded.

**CNN/DailyMail, sentiment-controlled vs. base:**

| Metric              | Base   | Sentiment-controlled |
|----------------------|--------|----------------------|
| ROUGE-1              | 0.407  | 0.394                |
| ROUGE-2              | 0.197  | 0.184                |
| ROUGE-L              | 0.311  | 0.302                |
| ROUGE-Lsum           | 0.342  | 0.360                |
| Sentiment alignment  | 0.899  | 0.946                |

Sentiment alignment (cosine similarity between the source article's and the
summary's sentiment vectors) improves meaningfully with sentiment control, at the
cost of a small drop in ROUGE-1/2/L (ROUGE-Lsum actually goes up slightly). Trained
on a 2,000-example slice for 1 epoch, a larger training run would likely narrow
this gap further.

**Limitation:** on manual inspection, the controlled model doesn't always represent
every tagged sentiment segment in its output: in one test article with three
tagged sections (negative, neutral, positive), the generated summary drew from the
negative and neutral segments but dropped the positive one. So sentiment control
shifts the model's behavior in aggregate, but doesn't guarantee every tag is
reflected in every summary.

## Project structure

Read in this order to understand the code:

- **`sentiment_utils.py`**: shared module. `tag_text_with_sentiment_prefixes`
  groups text into 3-sentence chunks and prefixes each with its dominant sentiment
  ([NEGATIVE]/[NEUTRAL]/[POSITIVE]); `calculate_sentiment_preservation` scores how
  well a summary preserves the source's sentiment profile. Used by both the data
  prep and evaluation steps below, so the tagging logic can't drift out of sync.
- **`bart_finetuning.py`**: fine-tunes BART on CNN/DailyMail directly. Kept as a
  documented negative result (see above); not part of the final pipeline.
- **`sentiment_tagging.ipynb`**: builds the sentiment-tagged training data from
  CNN/DailyMail, using `sentiment_utils`.
- **`bart_sentiment_finetuning.py`**: fine-tunes BART on the sentiment-tagged
  data, with `[POSITIVE]/[NEGATIVE]/[NEUTRAL]` registered as special tokens. If the
  tagged data or base model aren't found locally, it generates/downloads them.
- **`evaluate.ipynb`**: compares the base model against the sentiment-controlled
  model: ROUGE, sentiment alignment, a side-by-side comparison table, and one
  qualitative example.

## Models

- `models/bart_base_cnn/` — `facebook/bart-large-cnn`, unmodified.
- `models/bart_sentiment_controlled/` — BART fine-tuned on sentiment-tagged input.

## Data

- `data/tagged_train_nltk/`, `data/tagged_val_nltk/` — CNN/DailyMail articles
  (2,000 train / 200 validation) with sentiment-tag prefixes, built by
  `sentiment_tagging.ipynb`.

## Reproducing

1. `sentiment_tagging.ipynb` — builds the tagged datasets.
2. `bart_sentiment_finetuning.py` — trains the sentiment-controlled model
   (self-sufficient: tags data and downloads the base model if either is missing).
3. `evaluate.ipynb` — runs the comparison and produces the numbers above.

## Possible next steps

- A FastAPI endpoint accepting raw article text and returning a summary.
- Train on more than 2,000 examples / more than 1 epoch, to see whether the
  ROUGE gap narrows.
- A tagging scheme less prone to dropping a sentiment segment entirely.