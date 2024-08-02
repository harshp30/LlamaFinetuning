import gc
import torch
import config
from data_preperation import load_and_prepare_dataset
from model_setup import setup_model_and_tokenizer
from training import train_model
from inference import run_inference

def main():
    # Load and prepare dataset
    dataset = load_and_prepare_dataset(config.dataset_name)

    # Set up model, tokenizer, and LoRA configuration
    model, tokenizer, peft_config = setup_model_and_tokenizer(config)

    # Train the model
    trainer = train_model(model, dataset, peft_config, tokenizer, config)

    # Clear CUDA cache
    torch.cuda.empty_cache()
    gc.collect()

    # Save the trained model
    trainer.model.save_pretrained(config.new_model)

    # Run inference (example)
    prompt = "What is a large language model?"
    result = run_inference(trainer.model, tokenizer, prompt)
    print(result)

if __name__ == "__main__":
    main()