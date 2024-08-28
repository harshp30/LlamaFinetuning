from datasets import load_dataset

def load_and_prepare_dataset(dataset_name):
    """
    Load and prepare the dataset for training.
    """
    dataset = load_dataset(dataset_name, split="train")
    return dataset