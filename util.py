from transformers.optimization import Adafactor, AdamW, get_scheduler
from typing import Any, Dict, Union
import math
import torch
from datasets import load_metric
import loralib as lora

def customized_lora(model, all_rank, memory):
    def findMaxLowerPowerOf2(n):
        power = math.floor(math.log2(n))
        return 1 << (power - 1)

    def alg(all_rank, max_len):
        ans = list()
        while all_rank > 2:
            ans.append(findMaxLowerPowerOf2(all_rank))
            all_rank -= ans[-1]
            if len(ans) == max_len:
                return ans
        if all_rank == 2:
            ans.append(all_rank) 
        return ans
    ranks = alg(all_rank, 6)
    # print(f"ranks --> {ranks}")
    layer_rank = dict()
    target_attn_matrix = dict()
    target_ffn_matrix = dict()

    last_layer_idx = 11
    for idx, r in enumerate(ranks):
        layer_rank[str(last_layer_idx - idx)] = r
        target_attn_matrix[str(last_layer_idx - idx)] = ["query", "key", "value", "output"]
        target_ffn_matrix[str(last_layer_idx - idx)] = ["intermediate", "output"]

    only_lora_B = False
    for layer in target_attn_matrix.keys():
        for matrix in target_attn_matrix[layer]:
            rank = layer_rank[layer]
            alpha = 2 * rank
            # set attention.output
            if matrix == "output":
                module = model._modules["base_model"]._modules["encoder"]._modules["layer"]._modules[layer]._modules["attention"]._modules[matrix]._modules["dense"]
                lora_layer = lora.Linear(in_features=module.in_features, out_features=module.out_features, r=rank, lora_alpha=alpha)
                if only_lora_B:
                    lora_layer.lora_A.requires_grad = False
                lora_layer.weight = module.weight
                lora_layer.bias = module.bias
                model._modules["base_model"]._modules["encoder"]._modules["layer"]._modules[layer]._modules["attention"]._modules[matrix]._modules["dense"] = lora_layer
            else:
                module = model._modules["base_model"]._modules["encoder"]._modules["layer"]._modules[layer]._modules["attention"]._modules["self"]._modules[matrix]
                lora_layer = lora.Linear(in_features=module.in_features, out_features=module.out_features, r=rank, lora_alpha=alpha)
                if only_lora_B:
                    lora_layer.lora_A.requires_grad = False
                lora_layer.weight = module.weight
                lora_layer.bias = module.bias
                model._modules["base_model"]._modules["encoder"]._modules["layer"]._modules[layer]._modules["attention"]._modules["self"]._modules[matrix] = lora_layer
            

    for layer in target_ffn_matrix.keys():
        for matrix in target_ffn_matrix[layer]:
            rank = layer_rank[layer]
            module = model._modules["base_model"]._modules["encoder"]._modules["layer"]._modules[layer]._modules[matrix]._modules["dense"]
            lora_layer = lora.Linear(in_features=module.in_features, out_features=module.out_features, r=rank, lora_alpha=alpha)
            if only_lora_B:
                lora_layer.lora_A.requires_grad = False
            lora_layer.weight = module.weight
            lora_layer.bias = module.bias
            model._modules["base_model"]._modules["encoder"]._modules["layer"]._modules[layer]._modules[matrix]._modules["dense"] = lora_layer
    lora.mark_only_lora_as_trainable(model)
    
    return set_trainble_para(model, memory)

def set_trainble_para(model, memory):
    # set layers according to memory
    if memory == 2:
        for layer, para in model.named_parameters():
            if "11" in layer:
                if "lora" in layer:
                    para.requires_grad = True
                else:
                    para.requires_grad = False
            else:
                para.requires_grad = False
    elif memory == 4:
        for layer, para in model.named_parameters():
            if "10" in layer or "11" in layer:
                if "lora" in layer:
                    para.requires_grad = True
                else:
                    para.requires_grad = False
            else:
                para.requires_grad = False
    elif memory == 6:
        for layer, para in model.named_parameters():
            if "8" in layer or "9" in layer or "10" in layer or "11" in layer:
                if "lora" in layer:
                    para.requires_grad = True
                else:
                    para.requires_grad = False
            else:
                para.requires_grad = False
    else:
        pass
    # 设置head可训练
    model._modules["linear"].weight.requires_grad = True
    model._modules["linear"].bias.requires_grad = True
    
    return model

def add_adapter(model, width = 32, depth = 12):
    def make_only_adapter_trainable(model):
        for layer, para in model.named_parameters():
            if "adapter" in layer:
                para.requires_grad = True
            else:
                para.requires_grad = False

    from torch import nn
    class Adapter(nn.Module):
        def __init__(self, input_dim, bottleneck_dim):
            super().__init__()
            self.down_project = nn.Linear(input_dim, bottleneck_dim, bias=False) 
            self.activation = nn.ReLU()  
            self.up_project = nn.Linear(bottleneck_dim, input_dim, bias=False)
            
        def forward(self, x):
            x = self.down_project(x) 
            x = self.activation(x)
            x = self.up_project(x)
            return x
        
    layers = [str(l) for l in range(11, 11 - depth, -1)]
    for layer in layers:
        origin_layer = model._modules["base_model"]._modules["encoder"]._modules["layer"]._modules[layer]._modules["output"]._modules["LayerNorm"]
        from torch.nn import Sequential
        import copy
        new_layer = Sequential()
        new_layer.add_module(layer, copy.deepcopy(origin_layer))
        adapter = Adapter(input_dim=768, bottleneck_dim=width)
        new_layer.add_module('adapter', adapter)

        model._modules["base_model"]._modules["encoder"]._modules["layer"]._modules[layer]._modules["output"]._modules["LayerNorm"] = new_layer

    make_only_adapter_trainable(model)
    # 设置head可训练
    model._modules["linear"].weight.requires_grad = True
    model._modules["linear"].bias.requires_grad = True

    return model


def vallina_lora(model, depth = 12, rank = 8, alpha = 32, test_target_matrix=None):
    ####################################################
    ranks = [rank for _ in range(depth)]
    # print(f"ranks --> {ranks}")
    layer_rank = dict()
    target_attn_matrix = dict()
    target_ffn_matrix = dict()

    last_layer_idx = model._modules["base_model"].config.num_hidden_layers - 1
    
    if test_target_matrix is None:
        for idx, r in enumerate(ranks):
            layer_rank[str(last_layer_idx - idx)] = r
            if "deberta" in model._modules["base_model"].config.name_or_path:
                target_attn_matrix[str(last_layer_idx - idx)] = ["query_proj", "key_proj", "value_proj", "output"]
                target_ffn_matrix[str(last_layer_idx - idx)] = ["intermediate", "output"]
            elif "bert" in model._modules["base_model"].config.name_or_path:
                target_attn_matrix[str(last_layer_idx - idx)] = ["query", "key", "value", "output"]
                target_ffn_matrix[str(last_layer_idx - idx)] = ["intermediate", "output"]
            elif "tiny-llama" in model._modules["base_model"].config.name_or_path:
                target_attn_matrix[str(last_layer_idx - idx)] = ["q_proj", "k_proj", "v_proj", "o_proj"]
                target_ffn_matrix[str(last_layer_idx - idx)] = ["up_proj", "down_proj"]
            
    else:
        for idx, r in enumerate(ranks):
            layer_rank[str(last_layer_idx - idx)] = r
            if test_target_matrix in ["query", "key", "value", "attn.output"]:
                if test_target_matrix == "attn.output":
                    test_target_matrix = "output"
                target_attn_matrix[str(last_layer_idx - idx)] = [test_target_matrix]
                test_target_matrix = "attn.output"
            elif test_target_matrix in ["intermediate", "ffn.output"]:
                if test_target_matrix == "ffn.output":
                    test_target_matrix = "output"
                target_ffn_matrix[str(last_layer_idx - idx)] = [test_target_matrix]
                test_target_matrix = "ffn.output"

    only_lora_B = False
    for layer in target_attn_matrix.keys():
        for matrix in target_attn_matrix[layer]:
            rank = layer_rank[layer]
            alpha = 2 * rank
            # set attention.output
            if "deberta" in model._modules["base_model"].config.name_or_path:
                if matrix == "output":
                    module = model._modules["base_model"]._modules["encoder"]._modules["layer"]._modules[layer]._modules["attention"]._modules[matrix]._modules["dense"]
                    lora_layer = lora.Linear(in_features=module.in_features, out_features=module.out_features, r=rank, lora_alpha=alpha)
                    if only_lora_B:
                        lora_layer.lora_A.requires_grad = False
                    lora_layer.weight = module.weight
                    lora_layer.bias = module.bias
                    model._modules["base_model"]._modules["encoder"]._modules["layer"]._modules[layer]._modules["attention"]._modules[matrix]._modules["dense"] = lora_layer
                else:
                    module = model._modules["base_model"]._modules["encoder"]._modules["layer"]._modules[layer]._modules["attention"]._modules["self"]._modules[matrix]
                    lora_layer = lora.Linear(in_features=module.in_features, out_features=module.out_features, r=rank, lora_alpha=alpha)
                    if only_lora_B:
                        lora_layer.lora_A.requires_grad = False
                    lora_layer.weight = module.weight
                    lora_layer.bias = module.bias
                    model._modules["base_model"]._modules["encoder"]._modules["layer"]._modules[layer]._modules["attention"]._modules["self"]._modules[matrix] = lora_layer
            elif "bert" in model._modules["base_model"].config.name_or_path:
                if matrix == "output":
                    module = model._modules["base_model"]._modules["encoder"]._modules["layer"]._modules[layer]._modules["attention"]._modules[matrix]._modules["dense"]
                    lora_layer = lora.Linear(in_features=module.in_features, out_features=module.out_features, r=rank, lora_alpha=alpha)
                    if only_lora_B:
                        lora_layer.lora_A.requires_grad = False
                    lora_layer.weight = module.weight
                    lora_layer.bias = module.bias
                    model._modules["base_model"]._modules["encoder"]._modules["layer"]._modules[layer]._modules["attention"]._modules[matrix]._modules["dense"] = lora_layer
                else:
                    module = model._modules["base_model"]._modules["encoder"]._modules["layer"]._modules[layer]._modules["attention"]._modules["self"]._modules[matrix]
                    lora_layer = lora.Linear(in_features=module.in_features, out_features=module.out_features, r=rank, lora_alpha=alpha)
                    if only_lora_B:
                        lora_layer.lora_A.requires_grad = False
                    lora_layer.weight = module.weight
                    lora_layer.bias = module.bias
                    model._modules["base_model"]._modules["encoder"]._modules["layer"]._modules[layer]._modules["attention"]._modules["self"]._modules[matrix] = lora_layer
            
            elif "tiny-llama" in model._modules["base_model"].config.name_or_path:
                module = model._modules["base_model"]._modules["layers"]._modules[layer]._modules["self_attn"]._modules[matrix]
                lora_layer = lora.Linear(in_features=module.in_features, out_features=module.out_features, r=rank, lora_alpha=alpha)
                if only_lora_B:
                    lora_layer.lora_A.requires_grad = False
                lora_layer.weight = module.weight
                lora_layer.bias = module.bias
                model._modules["base_model"]._modules["layers"]._modules[layer]._modules["self_attn"]._modules[matrix] = lora_layer
            

    for layer in target_ffn_matrix.keys():
        for matrix in target_ffn_matrix[layer]:
            rank = layer_rank[layer]
            if "deberta" in model._modules["base_model"].config.name_or_path:
                module = model._modules["base_model"]._modules["encoder"]._modules["layer"]._modules[layer]._modules[matrix]._modules["dense"]
                lora_layer = lora.Linear(in_features=module.in_features, out_features=module.out_features, r=rank, lora_alpha=alpha)
                if only_lora_B:
                    lora_layer.lora_A.requires_grad = False
                lora_layer.weight = module.weight
                lora_layer.bias = module.bias
                model._modules["base_model"]._modules["encoder"]._modules["layer"]._modules[layer]._modules[matrix]._modules["dense"] = lora_layer
            elif "bert" in model._modules["base_model"].config.name_or_path:
                module = model._modules["base_model"]._modules["encoder"]._modules["layer"]._modules[layer]._modules[matrix]._modules["dense"]
                lora_layer = lora.Linear(in_features=module.in_features, out_features=module.out_features, r=rank, lora_alpha=alpha)
                if only_lora_B:
                    lora_layer.lora_A.requires_grad = False
                lora_layer.weight = module.weight
                lora_layer.bias = module.bias
                model._modules["base_model"]._modules["encoder"]._modules["layer"]._modules[layer]._modules[matrix]._modules["dense"] = lora_layer
            
            elif "tiny-llama" in model._modules["base_model"].config.name_or_path:
                module = model._modules["base_model"]._modules["layers"]._modules[layer]._modules["mlp"]._modules[matrix]
                lora_layer = lora.Linear(in_features=module.in_features, out_features=module.out_features, r=rank, lora_alpha=alpha)
                if only_lora_B:
                    lora_layer.lora_A.requires_grad = False
                lora_layer.weight = module.weight
                lora_layer.bias = module.bias
                model._modules["base_model"]._modules["layers"]._modules[layer]._modules["mlp"]._modules[matrix] = lora_layer
    lora.mark_only_lora_as_trainable(model)

    # 设置head可训练
    model._modules["linear"].weight.requires_grad = True
    model._modules["linear"].bias.requires_grad = True
    
    return model

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
            
def get_metrics(task:str):
    task_to_metric = {
    "cola": ["matthews_correlation", None],
    "sst2": ["accuracy", None],
    "mrpc": ["f1", "accuracy"],
    "stsb": ["pearsonr", None],
    "qqp": ["f1", "accuracy"],
    "mnli": ["accuracy", None],
    "mnli-mm": ["accuracy", None],
    "qnli": ["accuracy", None],
    "rte": ["accuracy", None],
    "wnli": ["accuracy", None],
    "ag_news": ["accuracy", None],
    "20news": ["accuracy", None]
    }
    from evaluate import load
    from datasets import load_metric
    metric = load(task_to_metric[task][0])
    metric_1 = load(task_to_metric[task][1]) if task_to_metric[task][1] else None
    return metric, metric_1

def compute_metrics(predictions, references, metric):
    if f"{metric.__class__.__name__ }" != 'Pearsonr':
        predictions = torch.argmax(predictions, dim=1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=references)


def prepare_inputs(inputs: Dict[str, Union[torch.Tensor, Any]], device) -> Dict[str, Union[torch.Tensor, Any]]:
    """
    Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
    handling potential state.
    """
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)
    return inputs


def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


def create_optimizer(model, adafactor=None, weight_decay=0.01, learning_rate=2e-5,
                     adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8):
    """
    Setup the optimizer.

    We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
    Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
    """
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]
    optimizer_cls = Adafactor if adafactor else AdamW
    if adafactor:
        optimizer_cls = Adafactor
        optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
    else:
        optimizer_cls = AdamW
        optimizer_kwargs = {
            "betas": (adam_beta1, adam_beta2),
            "eps": adam_epsilon,
        }
    optimizer_kwargs["lr"] = learning_rate
    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    return optimizer


def create_scheduler(optimizer, lr_scheduler_type: str="linear", num_training_steps: int=10, 
                     warmup_steps: int=0, warmup_ratio: float=0.0):
    """
    Setup the scheduler. The optimizer of the trainer must have been set up before this method is called.

    Args:
        num_training_steps (int): The number of training steps to do.
    """
    warmup_steps = (
        warmup_steps
        if warmup_steps > 0
        else math.ceil(num_training_steps * warmup_ratio)
    )
    lr_scheduler = get_scheduler(
        lr_scheduler_type,
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )
    return lr_scheduler