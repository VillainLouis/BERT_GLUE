from util import prepare_inputs, compute_metrics
from typing import Any, Dict, Union
import torch
from numpy import mean

def compute_loss(model, inputs):
    """
    
    """
    if "labels" in inputs:
        labels = inputs.pop("labels")
    outputs = model(**inputs)

    logits = outputs["logits"]
    loss = model.loss(logits, labels)
    metric, metric_1 = model.compute_metrics(predictions=logits, references=labels)
    
    return (loss, metric, metric_1)


def training_step(model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]],
                  optimizer: torch.optim.Optimizer) -> torch.Tensor:
    """
    Perform a training step on a batch of inputs.

    Subclass and override to inject custom behavior.

    Args:
        model (:obj:`nn.Module`):
            The model to train.
        inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
            The inputs and targets of the model.
        optimizer (torch.optim.Optimizer): 
            Optimizer instance for the training loop.
        lr_scheduler (torch.optim.lr_scheduler): 
            LR scheduler instance for the training loop.

            The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
            argument :obj:`labels`. Check your model's documentation for all accepted arguments.

    Return:
        :obj:`torch.Tensor`: The tensor with training loss on this batch.
    """
    loss, metric, metric_1 = compute_loss(model, inputs)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    model.zero_grad()
    # lr_scheduler.step()

    return loss.detach(), metric, metric_1


def eval_step(model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
    """
    Perform a training step on a batch of inputs.

    Subclass and override to inject custom behavior.

    Args:
        model (:obj:`nn.Module`):
            The model to train.
        inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
            The inputs and targets of the model.
        optimizer (torch.optim.Optimizer): 
            Optimizer instance for the training loop.
        lr_scheduler (torch.optim.lr_scheduler): 
            LR scheduler instance for the training loop.

            The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
            argument :obj:`labels`. Check your model's documentation for all accepted arguments.

    Return:
        :obj:`torch.Tensor`: The tensor with training loss on this batch.
    """
    model.eval()
    model.zero_grad()
    loss, metric, metric_1 = compute_loss(model, inputs)

    return loss.detach(), metric, metric_1


if __name__ == "__main__":
    
    import torch
    from util import *
    from dataloader import get_dataloader
    from train import training_step

    # Config Settings
    device = "cuda:0" # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_checkpoint="/data/jliu/models/bert-base-uncased"
    task = "rte"
    batch_size=32
    lr = 1e-4

    # Load DataLoader
    print(f"\nLoading data...")
    train_epoch_iterator = get_dataloader(task, model_checkpoint, "train", batch_size=batch_size)
    eval_epoch_iterator = get_dataloader(task, model_checkpoint, "validation", batch_size=batch_size)
    
    # Load Pre-trained Model
    from model import CustomBERTModel
    print(f"\nLoading pre-trained BERT model \"{model_checkpoint}\"")
    num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2
    model = CustomBERTModel(model_checkpoint, num_labels=num_labels, task=task)

    finetune_type = "heterlora"
    if finetune_type == "fedft":
        pass
    elif finetune_type == "fedlora":
        fedlora_depth = 12
        fedlora_rank = 16
        test_target_matrix = None
        model = vallina_lora(model, depth=fedlora_depth, rank=fedlora_rank, alpha=fedlora_rank * 2, test_target_matrix= test_target_matrix)
        
    elif finetune_type == "fedadapter":
        fedadpter_width = 32
        fedadpter_depth = 6
        model = add_adapter(model, width=fedadpter_width, depth=fedadpter_depth)

    elif finetune_type == "our":
        our_total_rank = 192
        memory = 8
        model = customized_lora(model,our_total_rank, memory)
    elif finetune_type == "heterlora":
        # 不同的秩测试
        client_rank = 64
        model = vallina_lora(model, rank=client_rank, alpha=client_rank * 2)
    else:
        raise NotImplementedError

    model.to(device)
    for layer, para in model.named_parameters():
        if para.requires_grad:
            print(f"{layer} --> {para.shape}")

    Optimizer=torch.optim.SGD(model.parameters(),lr)
    
    # Training Loop
    print(f"\nTraining begins in batches of {batch_size}..")
    Epoch = 10
    model.zero_grad()
    Optimizer.zero_grad()
    for ep in range(Epoch):
        # training
        iterator = iter(train_epoch_iterator)
        print(f"################ Epoch {ep} #####################")
        model.train()
        loss_all=[]
        metric_name = model.metric.name
        metric_1_name = None if model.metric_1 is None else model.metric_1.name
        metric_all=[]
        metric_1_all = []
        import time
        for step in range(len(train_epoch_iterator)):
            pre_time = time.time()
            inputs = prepare_inputs(next(iterator), device)
            step_loss, step_metric, step_metric_1 = training_step(model, inputs, Optimizer)
            cur_time = time.time()
            print(f"time per batch (with batch size = {batch_size}) --> {cur_time - pre_time}")
            pre_time = cur_time

            loss_all.append(step_loss.item())
            metric_all.append(step_metric[model.metric.name])
            if model.metric_1 is not None: 
                metric_1_all.append(step_metric_1[step_metric[model.metric_1.name]])
            
        # evaluation
        # iterator = iter(eval_epoch_iterator)
        # trange = range(len(eval_epoch_iterator))
        # model.eval()
        # loss_all=[]
        # metric_name = model.metric.name
        # metric_1_name = None if model.metric_1 is None else model.metric_1.name
        # metric_all=[]
        # metric_1_all = []
        # for step in tqdm(trange):
        #     inputs = prepare_inputs(next(iterator), device)
        #     step_loss, step_metric, step_metric_1 = eval_step(model, inputs)
        #     loss_all.append(step_loss.item())
        #     metric_all.append(step_metric[model.metric.name])
        #     if model.metric_1 is not None: 
        #         metric_1_all.append(step_metric_1[step_metric[model.metric_1.name]])
            
        # print("test loss:",mean(loss_all))
        # print(f"test {model.metric.name} --> {mean(metric_all)} ")
        # if model.metric_1 is not None:
        #     print(f"test {model.metric_1.name} -->  {mean(metric_1_all)}")
