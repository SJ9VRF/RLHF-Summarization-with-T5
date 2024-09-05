# Data processing and utility functions

import functools
from transformers import AutoTokenizer
from datasets import load_dataset


def load_datasets(dataset_name="xsum", splits=None):
    """
    Load the dataset with specified splits.
    :param dataset_name: Name of the dataset (default: "xsum")
    :param splits: List of splits to load (default: ["train", "validation", "test"])
    :return: Tuple containing loaded datasets.
    """
    if splits is None:
        splits = ["train", "validation", "test"]
    
    dataset = load_dataset(dataset_name, split=splits)
    return dataset


def tokenize_data(tokenizer, examples, remove_target=False, max_input_length=1024, max_output_length=128):
    """
    Tokenizes input and output texts.
    :param tokenizer: Tokenizer object (from transformers)
    :param examples: Dataset examples to tokenize
    :param remove_target: Whether to remove the target summaries (default: False)
    :param max_input_length: Maximum input token length (default: 1024)
    :param max_output_length: Maximum output token length (default: 128)
    :return: Tokenized input (and optionally, output) examples.
    """
    inputs = [doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    
    # Optionally remove the target summaries (used for inference)
    if not remove_target:
        labels = tokenizer(examples["summary"], max_length=max_output_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs


def prepare_tokenized_datasets(dataset, tokenizer, remove_target=False):
    """
    Tokenizes an entire dataset.
    :param dataset: Dataset to tokenize
    :param tokenizer: Tokenizer object (from transformers)
    :param remove_target: Whether to remove the target summaries (default: False)
    :return: Tokenized dataset
    """
    tokenize_fn = functools.partial(tokenize_data, tokenizer, remove_target=remove_target)
    tokenized_dataset = dataset.map(tokenize_fn, batched=True)
    return tokenized_dataset


def group_texts(examples, block_size=128):
    """
    Groups tokenized texts into blocks of a specified size.
    This is useful when dealing with long-form text generation tasks.
    :param examples: The examples to group
    :param block_size: Block size to use for grouping (default: 128)
    :return: Grouped examples
    """
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result
