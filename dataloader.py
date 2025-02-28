from torch.utils.data.dataloader import DataLoader
from datasets import load_dataset
from transformers.data.data_collator import DataCollatorWithPadding
from transformers import BertTokenizerFast, AutoTokenizer


def get_dataloader(task:str, model_checkpoint:str, split:str, dataloader_drop_last:bool=True, shuffle:bool=False,
                   batch_size:int=16, dataloader_num_workers:int=0, dataloader_pin_memory:bool=True) -> DataLoader:
    """To create encoded dataset dataloader for a given GLUE task.

    Args:
        task (str): GLUE task.
        model_checkpoint (str): tokenizer restoring model_checkpoint.
        split (str): "train", "validation", "test".
        dataloader_drop_last (bool, optional): Whether to drop the last incomplete batch (if the length of the dataset is not divisible by the batch size) or not. Defaults to True.
        batch_size (int): Number of samples in each batch.
        dataloader_num_workers (int, optional): Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded in the main process. Defaults to 0.
        dataloader_pin_memory (bool, optional): Whether you want to pin memory in data loaders or not. Defaults to True.

    Returns:
        dataloader(DataLoader): A toknized and encoded dataloader.
    """
    task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "ag_news": ("text", None),
    "20news": ("text", None),
    }

    sentence1_key, sentence2_key = task_to_keys[task]
    
    def preprocess_function(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], truncation=True, max_length=512, padding=True)
        return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True,  max_length=512, padding=True)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    if "tiny-llama" in model_checkpoint:
        tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    actual_task = "mnli" if task == "mnli-mm" else task
    if task == "ag_news":
        dataset = dataset = load_dataset("ag_news")
    elif task == "20news":
        dataset = load_dataset("SetFit/20_newsgroups")
    else:
        dataset = load_dataset("glue", actual_task)
    encoded_dataset = dataset.map(preprocess_function, batched=True)
    
    columns_to_return = ['input_ids', 'label', 'attention_mask']
    encoded_dataset.set_format(type='torch', columns=columns_to_return)

    if (split == "validation" or split == "test") and task == "mnli":
        split = "validation_matched"
    if (split == "validation" or split == "test") and task == "mnli-mm":
        split = "validation_mismatched"
    
    print(encoded_dataset)
    # from torch.utils.data import Subset
    # encoded_dataset[split] = Subset(encoded_dataset[split], range(int(0.1 * len(encoded_dataset[split]))))
    dataloader = DataLoader(
                    encoded_dataset[split],
                    shuffle=shuffle,
                    batch_size=batch_size,
                    collate_fn=data_collator,
                    drop_last=dataloader_drop_last,
                    num_workers=dataloader_num_workers,
                    pin_memory=dataloader_pin_memory,
    )
    
    return dataloader