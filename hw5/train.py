#!/usr/bin/env python

import bitsandbytes as bnb
from datasets import load_dataset
import os
from pathlib import Path
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

base_path = Path('/home/paperspace/llm-engineering-course/hw5')
model_path = base_path / "model"

os.environ["CUDA_VISIBLE_DEVICES"]="0"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_double_quant=True,
    bnb_4bit_compute_dtype='float32'
)

model_id = "NousResearch/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='right')

tokenizer.pad_token = tokenizer.eos_token

dataset_name = "mosaicml/instruct-v3"
dataset = load_dataset(dataset_name)
dataset["test"] = dataset["test"].select(range(50))


def create_prompt(sample):
  bos_token = "<s>"
  original_system_message = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
  system_message = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
  input = sample["prompt"].replace(original_system_message, "").replace("\n\n### Instruction\n", "").replace("\n### Response\n", "").strip()
  response = sample["response"]
  eos_token = "</s>"

  full_prompt = ""
  full_prompt += bos_token
  full_prompt += "### Instruction:"
  full_prompt += "\n" + system_message
  full_prompt += "\n\n### Input:"
  full_prompt += "\n" + input
  full_prompt += "\n\n### Response:"
  full_prompt += "\n" + response
  full_prompt += eos_token

  return full_prompt

model.config.use_cache = False
model = prepare_model_for_kbit_training(model)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


lora_alpha = 16 
lora_dropout = 0.1
lora_r = 64 

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)
print_trainable_parameters(model)

args = TrainingArguments(
    output_dir = base_path,
    # num_train_epochs=5,
    max_steps = 500, # comment out this line if you want to train in epochs
    per_device_train_batch_size = 16,
    warmup_steps =0.03,
    logging_steps=10,
    save_strategy='epoch',
    # evaluation_strategy="epoch",
    evaluation_strategy="steps",
    eval_steps=20, # comment out this line if you want to evaluate at the end of each epoch
    learning_rate=2e-4,
    bf16=True,
    lr_scheduler_type='constant',
)


max_seq_length = 2048

trainer = SFTTrainer(
    model=model,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    formatting_func=create_prompt,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)

trainer.train()

model.save_pretrained("model_path")
