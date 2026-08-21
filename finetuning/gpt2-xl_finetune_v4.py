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

from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig

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

version=2 # model version to use when saving

path_to_nlu_dir = "./"
data_dir = path_to_nlu_dir+""

train_path = "SATACT_v3_trn.jsonl"
val_path = "SATACT_v3_dev.jsonl"

data_name = 'SATACT'
save_dir = path_to_nlu_dir+"ft_Results/"

if not os.path.exists(save_dir):
  os.makedir(save_dir)

model_name = "openai-community/gpt2-xl" #"openai-community/gpt2-medium" #"openai-community/gpt2" #"meta-llama/Llama-3.2-1B"

# number of layers to train (taken from the last layers)
# if none, trains all layers
NUM_TRAINING_LAYERS = 2

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.truncation_side = 'left' # truncates from beginning

# setting max_new_tokens = 0
gen_config = GenerationConfig.from_pretrained(model_name)
gen_config.max_new_tokens = 0

# QUANTIZATION (4bit)
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
)

def init_model():
  model = AutoModelForCausalLM.from_pretrained(model_name,
                                               #quantization_config=bnb_config,
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
    # attention, fully connected, projection, positional embeddings, token embeddings
    lora_modules = ['c_attn', 'c_fc', 'c_proj', 'wpe', 'wte']
  elif 'llama' in model_name:
    lora_modules = ['q_proj', 'v_proj', 'k_proj', 'o_proj']
  
  r = 64
  config = LoraConfig(
        r=r,
        lora_alpha=r,
        lora_dropout=0.02,
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
  tokenized_input = tokenizer(pqa, padding='max_length', truncation=True, max_length=512) # increase max_length

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

    batch_size, num_choices, seq_len = input.shape
    input_ids_flat = input.reshape(batch_size*num_choices, -1)
    logits = logits.reshape(batch_size*num_choices, seq_len, -1)

    # Shift logits and target
    logits = logits[:, :-1, :]         # shape: (batch_size * 4, seq_len-1, vocab)
    targets = input_ids_flat[:, 1:]    # shape: (batch_size * 4, seq_len-1)

    # per choice loss
    per_token_loss = loss_fn(logits.reshape(-1, logits.size(-1)), # shape: (batch_size * 4 * seq_len-1, vocab)
                             targets.reshape(-1))                 # shape: (batch_size * 4 * seq_len-1)
    per_token_loss = per_token_loss.reshape(batch_size, num_choices, -1)  # shape: (batch_size, 4, seq_len-1)

    per_choice_loss = per_token_loss.sum(dim=-1) # shape: (batch_size, 4)

    preds = torch.argmin(per_choice_loss, dim=-1)

  acc = accuracy.compute(predictions=preds, references=labels)

  return acc

training_args = TrainingArguments(
    output_dir=f"{save_dir}checkpoints/",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    fp16=True,
    eval_strategy="epoch",
    eval_accumulation_steps=1,
    save_total_limit=1,
    logging_strategy='epoch',
    include_for_metrics = ["inputs"],
    prediction_loss_only=True
)

class CustomTrainer(Trainer):
  def compute_loss(self, model, inputs, 
                   return_outputs=False,
                   num_items_in_batch=None):
    input_ids = inputs["input_ids"]         # shape: (batch_size, 4, seq_len)
    attention_mask = inputs["attention_mask"]
    labels = inputs["labels"]               # shape: (batch_size,)

    batch_size, num_choices, seq_len = input_ids.shape
    input_ids = input_ids.reshape(batch_size * num_choices, seq_len)
    attention_mask = attention_mask.reshape(batch_size * num_choices, seq_len)

    # flatten to single batch dimension
    input_ids_flat = input_ids.reshape(batch_size*num_choices, -1)
    attention_mask_flat = attention_mask.reshape(batch_size*num_choices, -1)

    # Forward pass
    outputs = model(input_ids=input_ids_flat, attention_mask=attention_mask_flat)
    logits = outputs.logits  # shape: (batch_size * 4, seq_len, vocab)

    # Shift logits and target
    logits = logits[:, :-1, :]         # shape: (batch_size * 4, seq_len-1, vocab)
    targets = input_ids_flat[:, 1:]    # shape: (batch_size * 4, seq_len-1)

    # per choice loss
    per_token_loss = loss_fn(logits.reshape(-1, logits.size(-1)), # shape: (batch_size * 4 * seq_len-1, vocab)
                             targets.reshape(-1))                 # shape: (batch_size * 4 * seq_len-1)
    per_token_loss = per_token_loss.reshape(batch_size, num_choices, -1)  # shape: (batch_size, 4, seq_len-1)

    per_choice_loss = per_token_loss.sum(dim=-1) # shape: (batch_size, 4)

    # overall loss (smaller loss is better)
    total_loss_fn = nn.CrossEntropyLoss() # applies mean reduction by default (across batch)
    final_loss = total_loss_fn(-per_choice_loss, labels)

    return (final_loss, outputs) if return_outputs else final_loss

trainer = CustomTrainer(
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    model_init = init_model
)

def hyperparameter_search_settings():

    def optuna_hp_space(trial):
        return {
            "learning_rate": trial.suggest_categorical("learning_rate", [1e-3, 1e-4, 5e-5]),
            "num_train_epochs": trial.suggest_categorical("num_train_epochs", [2,3,5])
        }

    search_space = {'learning_rate': [1e-3, 1e-4, 5e-5],
                    'num_train_epochs': [2,3,5]}

    sampler=optuna.samplers.GridSampler(search_space)

    search_settings = {
        'backend' : 'optuna',
        'hp_space' : optuna_hp_space,
        'sampler' : sampler,
        'compute_objective' : lambda metrics: metrics['eval_loss'],
        'direction' : 'minimize',
        'n_trials' : 9,
    }
    return search_settings

if __name__ == "__main__":
  # hyperparameter search
  best = trainer.hyperparameter_search(**hyperparameter_search_settings())

  # load in best model with same PEFT/quantization
  # Note: uses last saved checkpoint from best parameter trial
  N = len(train_data)
  checkpoint_dir = os.path.join(save_dir, f"run-{best.run_id}", f"checkpoint-{N}")

  config = PeftConfig.from_pretrained(checkpoint_dir)
  base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,
                                                  #quantization_config=bnb_config,
                                                  device_map=None)
  model = PeftModel.from_pretrained(base_model, checkpoint_dir, torch_dtype=torch.float16)

  # push best model to hugging face
  model.push_to_hub(f"Salm00n/{model_name.split('/')[-1]}_{data_name}_v{version}")