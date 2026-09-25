import torch 
import os 
import shutil
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, DataCollatorForSeq2Seq, TrainingArguments, logging


torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


SAVE_DIR = "./saved_bart_model"
OUTPUT_DIR = "./results"
MODEL_NAME= "facebook/bart-large-cnn"


if os.path.exists(os.path.join(SAVE_DIR,"config.json")):
    print("Model already downloaded locally")
    tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(SAVE_DIR)
else:
    print("Retrieving model from HF hub")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

model.to(device)

# Lodaing and tokenizing the data
dataset = load_dataset("knkarthick/samsum")


def tokenize_dataset(data):
    model_inputs = tokenizer(data['dialogue'], truncation= True, max_length= 512)
    labels= tokenizer(text_target= data['summary'], truncation= True, max_length= 512)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

token_data = dataset.map(tokenize_dataset, batched= True, remove_columns=dataset['train'].column_names)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = TrainingArguments(
    output_dir = OUTPUT_DIR, 
    num_train_epochs=1,                 # start small to check it runs, then raise
    warmup_steps=500,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    weight_decay=0.01,
    logging_steps=50,
    eval_strategy='steps',
    eval_steps=500,
    save_strategy='no',                 
    fp16=torch.cuda.is_available(),
    report_to='none',
    push_to_hub=False, # for now
)

trainer = Trainer(
    model = model,
    args = training_args,
    data_collator = data_collator, 
    train_dataset = token_data['train'],
    eval_dataset = token_data['validation']
)

trainer.train()

trainer.save_model("./bart_samsum_finetuned")
tokenizer.save_pretrained("./bart_samsum_finetuned")


# saving results locally from colab
shutil.make_archive("bart_samsum_finetuned", "zip", "bart_samsum_finetuned")

try:
    from google.colab import files
    files.download("bart_samsum_finetuned.zip")
except ImportError:
    print("Not running in Colab — zip saved locally at ./bart_samsum_finetuned.zip")