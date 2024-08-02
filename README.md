# LlamaFinetuning: Fine-tuning Llama 2 with QLoRA

**This project demonstrates the fine-tuning of the Llama 2 model using Quantized Low-Rank Adaptation (QLoRA) techniques. It showcases an efficient approach to adapting large language models for specific tasks while minimizing computational requirements.**

---

**Technical Topics:** *Natural Language Processing, Transfer Learning, Model Fine-tuning, Quantization, Low-Rank Adaptation*

**Tools / Technologies:** *Python, PyTorch, Transformers, PEFT, bitsandbytes, TRL (Transformer Reinforcement Learning)*

**Use Case:** *Fine-tuning large language models for specific tasks or domains with reduced computational resources*

---

### Technical Explanation:

#### Model Configuration -- [config.py](https://github.com/harshp30/LlamaFinetuning/blob/main/config.py)

This script defines the configuration parameters for the model, training process, and QLoRA setup.

1. **Model and Dataset Selection:**
   - Uses the "NousResearch/Llama-2-7b-chat-hf" model
   - Trains on the "mlabonne/guanaco-llama2-1k" dataset

2. **QLoRA Configuration:**
   - `lora_r = 32`: Rank of the LoRA update matrices
   - `lora_alpha = 16`: Scaling factor for LoRA
   - `lora_dropout = 0.1`: Dropout probability for LoRA layers

3. **Quantization Settings:**
   - Uses 4-bit quantization with NF4 type
   - Computes in float16 precision

4. **Training Parameters:**
   - 10 epochs with gradient accumulation over 8 steps
   - Learning rate of 2e-4 with cosine scheduler
   - Max sequence length of 256 tokens

---

#### Data Preparation -- [data_preparation.py](https://github.com/harshp30/LlamaFinetuning/blob/main/data_preparation.py)

Handles loading and preparing the dataset for fine-tuning.

1. **Dataset Loading:**
   - Utilizes the Hugging Face `datasets` library to load the specified dataset

2. **Data Processing:**
   - Returns the loaded dataset without additional preprocessing, leveraging the pre-processed nature of the chosen dataset

---

#### Model Setup -- [model_setup.py](https://github.com/harshp30/LlamaFinetuning/blob/main/model_setup.py)

Manages the initialization of the model, tokenizer, and LoRA configuration.

1. **Quantization Configuration:**
   - Sets up 4-bit quantization using `BitsAndBytesConfig`

2. **Model Initialization:**
   - Loads the pre-trained Llama 2 model with quantization settings
   - Prepares the model for k-bit training

3. **Tokenizer Setup:**
   - Initializes the tokenizer with appropriate padding settings

4. **LoRA Configuration:**
   - Configures LoRA with specified rank, alpha, and dropout
   - Targets only the query and value projection layers ("q_proj", "v_proj") for efficiency

---

#### Training Process -- [training.py](https://github.com/harshp30/LlamaFinetuning/blob/main/training.py)

Handles the fine-tuning process using the SFTTrainer from the TRL library.

1. **Training Arguments:**
   - Sets up training parameters including batch size, learning rate, and optimizer

2. **SFTTrainer Initialization:**
   - Utilizes SFTTrainer for efficient supervised fine-tuning
   - Configures max sequence length and packing settings

3. **Training Execution:**
   - Runs the training process with the specified configuration

---

#### Inference -- [inference.py](https://github.com/harshp30/LlamaFinetuning/blob/main/inference.py)

Provides functionality to run inference on the fine-tuned model.

1. **Pipeline Setup:**
   - Creates a text generation pipeline using the fine-tuned model and tokenizer

2. **Text Generation:**
   - Generates text based on input prompts
   - Limits output to a maximum of 200 tokens

---

#### Main Execution -- [main.py](https://github.com/harshp30/LlamaFinetuning/blob/main/main.py)

Orchestrates the entire fine-tuning process from data loading to inference.

1. **Process Flow:**
   - Loads and prepares the dataset
   - Sets up the model, tokenizer, and LoRA configuration
   - Executes the training process
   - Saves the fine-tuned model
   - Demonstrates inference with an example prompt

2. **Memory Management:**
   - Clears CUDA cache and performs garbage collection after training

---

### Key Fine-tuning Techniques:

1. **Quantized Low-Rank Adaptation (QLoRA):**
   - Combines quantization and Low-Rank Adaptation for efficient fine-tuning
   - Enables adaptation of large models with minimal memory overhead

2. **4-bit Quantization:**
   - Reduces model size and memory footprint significantly
   - Utilizes NF4 quantization type for balanced precision and efficiency

3. **Targeted LoRA:**
   - Applies LoRA only to specific layers (query and value projections)
   - Balances adaptation capability and computational efficiency

4. **Gradient Accumulation:**
   - Allows for larger effective batch sizes on limited hardware
   - Set to accumulate over 8 steps, enabling more stable updates

5. **Learning Rate Optimization:**
   - Uses a relatively high learning rate (2e-4) for LoRA parameters
   - Implements a cosine learning rate scheduler for controlled convergence

---

### Next Steps:

- Experiment with different LoRA configurations (e.g., varying rank or target modules)
- Implement evaluation metrics to quantify fine-tuning performance
- Explore task-specific datasets for more focused adaptation
- Investigate the impact of different quantization schemes on model performance

---

### Citations:

```
Finetuning Tutorial: https://youtu.be/iOdFUJiB0Zc?si=Q20zSl_NH3eoPE4C
Llama 2: https://ai.meta.com/llama/
QLoRA: https://arxiv.org/abs/2305.14314
PEFT: https://github.com/huggingface/peft
TRL: https://github.com/lvwerra/trl
```