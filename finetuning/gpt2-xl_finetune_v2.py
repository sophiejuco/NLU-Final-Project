from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import os

from collections import defaultdict

import torch
import torch.nn as nn
import transformers
import re

import optuna
import evaluate

import pickle

from peft import LoraConfig, get_peft_model

from datasets import Dataset
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          Trainer,
                          TrainingArguments,
                          DataCollatorForMultipleChoice,
                          BitsAndBytesConfig,
                          GenerationConfig)
                          
from huggingface_hub import login

login(token="")

print(torch.cuda.device_count())

# training data path

path_to_nlu_dir = "./"
data_dir = path_to_nlu_dir+""

train_path = "SATACT_v3_trn.jsonl"
val_path = "SATACT_v3_dev.jsonl"

data_name = 'SATACT'
save_dir = path_to_nlu_dir+"Results/ft_Results/"

model_name = "openai-community/gpt2" #"meta-llama/Llama-3.2-1B"

# number of layers to train (taken from the last layers)
# if none, trains all layers
NUM_TRAINING_LAYERS = 2

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# setting max_new_tokens = 0
gen_config = GenerationConfig.from_pretrained(model_name)
gen_config.max_new_tokens = 0

def init_model():
  # QUANTIZATION (4bit)
  bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
  )

  model = AutoModelForCausalLM.from_pretrained(model_name,
                                               quantization_config=bnb_config,
                                               device_map=None)

  # freeze params
  for param in model.parameters():
    param.requires_grad = False

  ''' currently not used
  if NUM_TRAINING_LAYERS is not None:
    # unfreeze last layer(s)
    for block in model.transformer.h[-NUM_TRAINING_LAYERS:]:
      for param in block.parameters():
        param.requires_grad = True

    # unfreeze final layer norm + output head
    for param in model.transformer.ln_f.parameters():
      param.requires_grad = True
    for param in model.lm_head.parameters():
      param.requires_grad = True
  '''

  # PEFT model
  if 'gpt' in model_name:
    lora_modules = ['c_attn', 'c_fc']
  elif 'llama' in model_name:
    lora_modules = ['q_proj', 'v_proj', 'k_proj', 'o_proj']
  config = LoraConfig(
        r=1,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=lora_modules,
        bias="none",
        task_type='CAUSAL_LM',
  )

  model = get_peft_model(model, config)
  model.generation_config = gen_config

  trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
  total = sum(p.numel() for p in model.parameters())
  print(f"Trainable params: {trainable} / {total} ({100 * trainable / total:.2f}%)\n")

  return model

data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)

def create_input(line, sys_prompt='', fs_demos=''):
  pqa = [f"{fs_demos}Q: {line['context']} {line['question']}\nA:{sys_prompt} {line[i]}"
          for i in ['answerA', 'answerB', 'answerC', 'answerD']]
  label = 'ABCD'.index(line['correct'])

  # tokenize input
  tokenized_input = tokenizer(pqa, padding='max_length', truncation=True, max_length=512)

  return {'input_ids': tokenized_input['input_ids'],
          'attention_mask': tokenized_input['attention_mask'],
          'label': label}

# training data
with open(data_dir + train_path, 'r') as f:
  train_data_raw = [json.loads(line) for line in f]

train_data = Dataset.from_list([create_input(x) for x in tqdm(train_data_raw)])

# val data
with open(data_dir + val_path, 'r') as f:
  val_data_raw = [json.loads(line) for line in f]

val_data = Dataset.from_list([create_input(x) for x in tqdm(val_data_raw)])

print(f'\nTraining Size = {len(train_data)}')
print(f'Validation Size = {len(val_data)}')

loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id,
                              reduction="none")
accuracy = evaluate.load('accuracy')

def compute_metrics(eval_pred):
  logits, labels, input = eval_pred

  # logits shape: (batch_size, num_choices, seq_len, vocab_size)
  # labels shape: (batch_size,)
  # input shape: (batch_size, num_choices, seq_len)

  with torch.no_grad():
    if isinstance(logits, np.ndarray):
      logits = torch.tensor(logits)
    if isinstance(labels, np.ndarray):
      labels = torch.tensor(labels)
    if isinstance(input, np.ndarray):
      input = torch.tensor(input)

    if 'llama' in model_name:
      batch_size, num_choices, seq_len = input.shape
      logits = logits.reshape(batch_size, num_choices, seq_len, -1)
  
    logits = logits[:, :, :-1, :]
    targets = input[:, :, 1:]

    # get prediction with CrossEntropyLoss
    loss = loss_fn(logits.permute(0, 3, 1, 2), targets)
    total_loss = torch.sum(loss, dim=-1)
    
    preds = torch.argmin(total_loss, dim=-1)

  return accuracy.compute(predictions=preds, references=labels)

training_args = TrainingArguments(
    output_dir=f"{save_dir}checkpoints/",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    learning_rate=5e-5, # REMOVE THIS LINE IF DOING HYPERPARAMETER TUNING
    num_train_epochs = 2,
    fp16=True,
    eval_strategy="epoch",
    eval_accumulation_steps=1,
    save_total_limit=2,
    logging_strategy='epoch',
    include_for_metrics = ["inputs"],
)

class CustomTrainer(Trainer):
  def compute_loss(self, model, inputs, 
                   return_outputs=False,
                   num_items_in_batch=None):
    input_ids = inputs["input_ids"]         # shape: (batch_size, 4, seq_len)
    attention_mask = inputs["attention_mask"]
    labels = inputs["labels"]               # shape: (batch_size,)

    if 'llama' in model_name:
      batch_size, num_choices, seq_len = input_ids.shape
      input_ids = input_ids.reshape(batch_size * num_choices, seq_len)
      attention_mask = attention_mask.reshape(batch_size * num_choices, seq_len)

    # Forward pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # shape: (batch_size, 4, seq_len, vocab)

    if 'llama' in model_name:
      logits = logits.reshape(batch_size, num_choices, seq_len, -1)
      input_ids = input_ids.reshape(batch_size, num_choices, seq_len)

    # Shift logits and target
    logits = logits[:, :, :-1, :]         # shape: (batch_size, 4, seq_len-1, vocab)
    targets = input_ids[:, :, 1:]         # shape: (batch_size, 4, seq_len-1)

    # per choice loss
    loss = loss_fn(logits.permute(0, 3, 1, 2), targets)    # shapes: (batch_size, vocab, 4, seq_len-1),
    total_loss = loss.sum(dim=-1)                          #         (batch_size, 4, seq_len-1)

    # overall loss (smaller loss is better)
    total_loss_fn = nn.CrossEntropyLoss()
    final_loss = total_loss_fn(total_loss, labels)

    return (final_loss, outputs) if return_outputs else final_loss

trainer = CustomTrainer(
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data.select(range(50)),
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    model_init = init_model
)

def hyperparameter_search_settings():

    def optuna_hp_space(trial):
        return {
            "learning_rate": trial.suggest_categorical("learning_rate", [1e-3, 1e-4, 3e-4, 3e-5])
        }

    search_space = {'learning_rate': [1e-3, 1e-4, 3e-4, 3e-5]}

    sampler=optuna.samplers.GridSampler(search_space)

    search_settings = {
        'backend' : 'optuna',
        'hp_space' : optuna_hp_space,
        'sampler' : sampler,
        'compute_objective' : lambda metrics: metrics['eval_accuracy'],
        'direction' : 'maximize',
        'n_trials' : 4,
    }
    return search_settings

if __name__ == "__main__":
  trainer.train()
  trainer.push_to_hub(f"Salm00n/{model_name}_{data_name}_v1")