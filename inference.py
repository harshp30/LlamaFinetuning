from transformers import pipeline

def run_inference(model, tokenizer, prompt):
    """
    Run inference on the trained model.
    """
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
    result = pipe(f"[INST] {prompt} [/INST]")
    return result[0]['generated_text']