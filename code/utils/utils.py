import os
import random
import numpy as np
import pandas as pd
import pathlib
import re
import json
import torch
import math
import evaluate


SEED = 21
BASE_DIR = "./data"


def get_num_training_steps(train_set, configs):
    
    return math.ceil(float(len(train_set)) / configs.batch_size) * configs.num_epochs


def clean_str(text):
    # Remove non breaking spaces (\u00A0), etc
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def save_csv(df, filename, dirname):
    pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
    filepath = os.path.join(dirname, f"{filename}.csv")
    df.to_csv(filepath, encoding='utf-8', index=False)


def load_dataset():
    folder = os.path.join(BASE_DIR, "dataset_complete")
    df = load_df("syllabus_qa_dataset", folder)

    return df


def load_dataset_post_filtering():
    df_train = load_df("train", "./data/dataset_split")
    df_val = load_df("val", "./data/dataset_split")
    df_test = load_df("test", "./data/dataset_split")

    return df_train, df_val, df_test


def load_df(filename, folder, nrows=None):
    filepath = os.path.join(folder, f"{filename}.csv")
    df = pd.read_csv(filepath, encoding="utf-8", nrows=nrows)
    
    return df


def load_json(filename, folder):
    filepath = os.path.join(folder, f"{filename}.json")
    with open(filepath) as f:
        data = json.load(f)
    
    return data


def save_json(data, filename, dirname):
    pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
    filepath = os.path.join(dirname, f"{filename}.json")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent = 4)


def merge_dict(dict_list):
    # https://stackoverflow.com/questions/3494906/how-do-i-merge-a-list-of-dicts-into-a-single-dict
    result = {}
    for d in dict_list:
        result.update(d)
    
    return result

    
def set_random_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sanitize_configs(configs):
    assert configs.prompt_style == 1 or configs.prompt_style == 2
    # Standalone testing
    if( configs.testing ):
        configs.log_wandb = False
    if( configs.debug ):
        configs.num_epochs = 1
        configs.log_wandb = False

    return configs


def get_device(configs):
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    if configs.use_cuda: 
        if torch.cuda.is_available():
            device = torch.device('cuda')
        assert device.type == 'cuda', 'Error: No GPU found'
    else:
        device = torch.device('cpu')  
    
    return device


def tonp(x):
    if isinstance(x, (np.ndarray, float, int)):
        return np.array(x)
    else:
        return x.detach().cpu().numpy()


def aggregate_metrics(outputs):    
    res = {}
    for k in outputs[0].keys():
        all_logs = np.concatenate([tonp(x[k]).reshape(-1) for x in outputs])
        res[k] = np.mean(all_logs)
            
    return res


def save_model(trainer, configs, wandb_run_name):
    checkpoint_dir = f"{configs.model_checkpoint_dir}/{configs.exp_name}/{wandb_run_name}/best_val_loss/lora_model"
    pathlib.Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    # Save tokenizer
    trainer.tokenizer.save_pretrained(checkpoint_dir)
    # Save LoRA model only, not complete model
    trainer.model.model.save_pretrained(checkpoint_dir)


def compute_rouge_l_f1(targets, predictions, configs):
    rouge_metric = evaluate.load("rouge", seed=configs.seed)
    scores = rouge_metric.compute(predictions=predictions, references=targets, use_stemmer=True, use_aggregator=False)["rougeL"]
    rouge_l_f1 = rouge_metric.compute(predictions=predictions, references=targets, use_stemmer=True, use_aggregator=True)["rougeL"]

    return rouge_l_f1, scores


def compute_accuracy(targets_question_types, preds_question_types):
    res = [(target_question_type == pred_question_type) for (target_question_type, pred_question_type) in zip(targets_question_types, preds_question_types)]
    accuracy = float(sum(res))/len(res)

    return accuracy