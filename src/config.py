import torch

# Model and dataset configuration
model_name = "NousResearch/Llama-2-7b-chat-hf"
dataset_name = "mlabonne/guanaco-llama2-1k"
new_model = "Llama-2-7b-chat-finetune"

# QLoRA parameters
lora_r = 32  # Reduced from 64
lora_alpha = 16
lora_dropout = 0.1

# bitsandbytes parameters
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

# Training parameters
output_dir = "./results"
num_train_epochs = 10
fp16 = False
bf16 = False
per_device_train_batch_size = 1
per_device_eval_batch_size = 1
gradient_accumulation_steps = 8 
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "cosine"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 0
logging_steps = 25

# SFT parameters
max_seq_length = 256
packing = False

# Device configuration
device_map = {"": 0}

# Compute dtype
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)