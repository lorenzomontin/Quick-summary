"""
Fine-tunes BART on sentiment-tagged CNN/DailyMail articles.
 
Input: the tagged datasets produced by sentiment_tagging.ipynb
       (./data/tagged_train_nltk, ./data/tagged_val_nltk), where each
       example's 'article_with_sentiment' field has 3-sentence chunks
       prefixed with [NEGATIVE]/[NEUTRAL]/[POSITIVE].
Output: ./models/bart_sentiment_controlled
 
Run this on a GPU (Colab T4 or similar) — CPU training of bart-large is
impractically slow.
"""
import gc
import os
import shutil
 
import torch
from datasets import load_dataset, load_from_disk
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, Trainer,
                          DataCollatorForSeq2Seq, TrainingArguments)
from sentiment_utils import add_sentiment_prefixes
 
torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
 
BASE_MODEL_DIR = "./models/bart_base_cnn"
MODEL_NAME = "facebook/bart-large-cnn"
OUTPUT_DIR = "./results_sentiment"
FT_DIR = "./models/bart_sentiment_controlled"
TRAIN_DATA_DIR = "./data/tagged_train_nltk"
VAL_DATA_DIR = "./data/tagged_val_nltk"
 
SPECIAL_TOKENS = ['[POSITIVE]', '[NEGATIVE]', '[NEUTRAL]']
 
# ==================== Free up VRAM before loading BART ====================
# If a sentiment classifier (e.g. from sentiment_utils) is still in memory
# from an earlier cell/session, clear it first — bart-large is heavy enough
# on a T4 that this matters.
gc.collect()
torch.cuda.empty_cache()
# ============================================================================
 
# Start from a fresh, unpolluted copy of the base model — not bart_cnn_finetuned,
# since that fine-tune made ROUGE worse and was abandoned.
if os.path.exists(os.path.join(BASE_MODEL_DIR, "config.json")):
    print("Loading base model locally from", BASE_MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_DIR)
else:
    print("Downloading base model from the Hub")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    # Save the plain base model now, before special tokens are added below —
    # evaluate.ipynb needs this exact folder to exist for the comparison.
    os.makedirs(BASE_MODEL_DIR, exist_ok=True)
    tokenizer.save_pretrained(BASE_MODEL_DIR)
    model.save_pretrained(BASE_MODEL_DIR)
 
# Register the sentiment control tokens so the tokenizer keeps them whole
# instead of splitting them into sub-word pieces, then resize the model's
# embedding matrix to match.
tokenizer.add_special_tokens({'additional_special_tokens': SPECIAL_TOKENS})
model.resize_token_embeddings(len(tokenizer))
model.to(device)
 
# ---- Load the pre-tagged datasets, tagging them now if they don't exist yet ----
# Same fallback pattern as the model above: don't assume a prior step ran on
# this machine, just produce what's missing.
if os.path.exists(TRAIN_DATA_DIR) and os.path.exists(VAL_DATA_DIR):
    print("Loading pre-tagged data from", TRAIN_DATA_DIR, "and", VAL_DATA_DIR)
    tagged_train = load_from_disk(TRAIN_DATA_DIR)
    tagged_val = load_from_disk(VAL_DATA_DIR)
else:
    print("Tagged data not found locally — tagging it now (this is the slow part)")
    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")
    train_subset = dataset['train'].select(range(2000))
    val_subset = dataset['validation'].select(range(200))
 
    tagged_train = train_subset.map(add_sentiment_prefixes)
    tagged_val = val_subset.map(add_sentiment_prefixes)
 
    tagged_train.save_to_disk(TRAIN_DATA_DIR)
    tagged_val.save_to_disk(VAL_DATA_DIR)
    print("Saved tagged data to", TRAIN_DATA_DIR, "and", VAL_DATA_DIR)
 
 
def tokenize_sentiment_dataset(data):
    model_inputs = tokenizer(data['article_with_sentiment'], truncation=True, max_length=1024)
    labels = tokenizer(text_target=data['highlights'], truncation=True, max_length=256)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs
 
 
tokenized_train = tagged_train.map(tokenize_sentiment_dataset, batched=True,
                                    remove_columns=tagged_train.column_names)
tokenized_val = tagged_val.map(tokenize_sentiment_dataset, batched=True,
                                remove_columns=tagged_val.column_names)
 
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
 
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    weight_decay=0.01,
    logging_steps=50,
    eval_strategy='steps',
    eval_steps=100,
    save_strategy='no',
    fp16=torch.cuda.is_available(),
    report_to='none',
    gradient_checkpointing=True,  # reduces VRAM footprint, important on a free T4
)
 
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
)
 
print("Starting sentiment-controlled fine-tuning...")
trainer.train()
 
trainer.save_model(FT_DIR)
tokenizer.save_pretrained(FT_DIR)
print("Saved fine-tuned model to", FT_DIR)
 
# ---- Download from Colab, if that's where this ran ----
shutil.make_archive("bart_sentiment_controlled", "zip", FT_DIR)
try:
    from google.colab import files
    files.download("bart_sentiment_controlled.zip")
except ImportError:
    print("Not running in Colab — zip saved locally at ./bart_sentiment_controlled.zip")
