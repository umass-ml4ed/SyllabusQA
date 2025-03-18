import torch
import hydra
import wandb
import time
from omegaconf import OmegaConf
from tqdm import tqdm
import gc

from code.utils.utils import set_random_seed, aggregate_metrics, save_model, sanitize_configs, get_device, get_num_training_steps
from code.utils.load_data import load_data, get_data_loaders
from code.finetune.model import Trainer
from code.finetune.test import test
from code.finetune.batch_collator import CollateWraperGenerative
from code.retrievalauggen.get_retrieved_text import get_retrieved_syllabi_chunks


def train(configs, device):
    # Load data
    train_set, val_set, test_set = load_data(configs)
    # Load retrieved syllabi chunks for RAG if applicable
    retrieved_syllabi_chunks = get_retrieved_syllabi_chunks(configs) if configs.rag else None
    # Load model
    num_training_steps = get_num_training_steps(train_set, configs)
    trainer = Trainer(configs, device, num_training_steps)
    # Get data loaders
    train_loader, val_loader = get_data_loaders(train_set, val_set, CollateWraperGenerative, trainer.tokenizer, device, configs, retrieved_syllabi_chunks)

    # Best validation metric
    best_val_metric = float("inf")

    # Training loop
    with tqdm(range(configs.num_epochs)) as tepoch:
        for cur_iter in tepoch:
            tepoch.set_description("Epoch {}".format(cur_iter))
            start_time = time.time()
            
            # Train epoch
            trainer.set_train_mode() # Set train mode for model
            train_logs = []
            with tqdm(train_loader, unit="batch", leave=False) as tbatch:
                for batch_num, batch in enumerate(tbatch):
                    tbatch.set_description("Batch {}".format(batch_num))
                    logs = trainer.train_step(batch)  
                    train_logs.append(logs)
                    if configs.log_wandb:
                        wandb.log({"logs/train/lr": logs["lr"]})
            
            # After every training epoch, push logs to weights and biases
            train_it_time = time.time() - start_time
            train_logs = aggregate_metrics(train_logs)
            if configs.log_wandb:
                wandb.log({"logs/train/it_time": train_it_time})
                wandb.log({"metrics/train/loss": train_logs['loss']})
                wandb.log({"logs/train/cur_iter" : cur_iter})

            # Evaluate on validation set after every training epoch
            val_logs, best_val_metric = validate(val_loader, best_val_metric, trainer, configs, cur_iter)

            # Update training tqdm progress bar
            tepoch.set_postfix({"train loss" : train_logs['loss'], "val loss" : val_logs['loss']})
    
    if( configs.run_testing ):
        # Test with best saved model
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        test(test_set, wandb.run.name, configs, device)


def validate(val_loader, best_val_metric, trainer, configs, cur_iter):
    # Evaluation epoch
    # Set eval mode for model
    trainer.set_eval_mode()
    val_logs = []
    eval_start_time = time.time()
    for batch in val_loader:
        logs = trainer.val_step(batch)
        val_logs.append(logs)
    eval_it_time = time.time()-eval_start_time

    # Aggregate logs across batches
    val_logs = aggregate_metrics(val_logs)
    # Update metrics and save model
    if( float(val_logs["loss"]) < best_val_metric ):
        best_val_metric = float(val_logs["loss"])
        # Save model with best validation loss
        save_model(trainer, configs, wandb.run.name)
        if configs.log_wandb:
            wandb.log({"logs/val/best_loss_epoch": cur_iter})
    # Push logs to weights and biases
    if configs.log_wandb:
        wandb.log({"metrics/val/loss": val_logs['loss']})
        wandb.log({"metrics/val/best_loss": best_val_metric})
        wandb.log({"logs/val/it_time": eval_it_time})
    
    return val_logs, best_val_metric


@hydra.main(version_base=None, config_path=".", config_name="configs")
def main(configs):
    # Make reproducible
    set_random_seed(configs.seed)
    # Sanitize configs
    configs = sanitize_configs(configs)
    # Get device
    device = get_device(configs)
    # Link training to weights and biases
    if configs.log_wandb:
        wandb.init(project=configs.wandb_project)
        wandb.config.update(OmegaConf.to_container(configs, resolve=True))
    # Train model
    train(configs, device)


if __name__ == '__main__':
    main()