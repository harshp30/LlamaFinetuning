import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training

def setup_model_and_tokenizer(config):
    """
    Set up the model, tokenizer, and LoRA configuration.
    """
    # Set up quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.use_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=config.compute_dtype,
        bnb_4bit_use_double_quant=config.use_nested_quant,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map=config.device_map,
        trust_remote_code=True,
        use_cache=False,
    )

    # Prepare the model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Set up LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        r=config.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],  # Only target specific modules
    )

    return model, tokenizer, peft_config