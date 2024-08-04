import os
import shutil
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

import wandb
import pandas as pd
import matplotlib.pyplot as plt

from dataset import get_data_loader
from loss import InstanceLoss, FeatureLoss
from modules.network import Network
from utils.general_utils import load_config, set_seed, get_experiment_name, count_parameters, save_best_model, move_model_to_finished
from utils.wandb_utils import init_wandb
from utils.metrics import evaluate
from utils.visualization import visualize_embeddings
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.experiment_records import ExperimentRecords
    

def run(config):
    set_seed(config["seed"])
    experiment_name = get_experiment_name(config)
    config["project"] = f"{config['project']}-{config['dataset']}"
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    # Check if 'mps' backend is available for Apple Silicon support
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        
    train_loader, test_loader, visualize_loader = get_data_loader(config)
    model = Network(config["backbone"], config['feature_num'], config["hidden_dim"]).to(device)
    print(f'The model has {count_parameters(model):,} trainable parameters.')

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config['weight_decay'])
    
    if config["use_scheduler"]:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    else:
        scheduler = None

    instance_loss_fn = InstanceLoss(config["batch_size"], config["instance_temperature"], device=device)
    feature_loss_fn = FeatureLoss(config['feature_num'], config["feature_temperature"], device=device)

    records = ExperimentRecords()
    best_nmi = 0.0
    
    start_epoch = 1
    run_id = None
    checkpoint_path = f'checkpoints/{config["dataset"]}_checkpoint_{experiment_name}.pth.tar'
    
    if config["reload"] and os.path.exists(checkpoint_path):
        start_epoch, run_id, best_nmi = load_checkpoint(model, optimizer, records, filename=checkpoint_path, scheduler=scheduler)
        
    init_wandb(config, experiment_name, run_id)
        
    for epoch in range(start_epoch, config["epochs"]+1):
        model.train()
        instance_epoch_loss = 0.0
        feature_epoch_loss = 0.0
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for i, (x1, x2, _) in enumerate(progress_bar):
            x1, x2 = x1.to(device), x2.to(device)
            
            optimizer.zero_grad()

            z1, z2, f1, f2 = model(x1, x2)

            instance_loss = instance_loss_fn(z1, z2)
            feature_loss = feature_loss_fn(f1, f2)
                
            loss = instance_loss + feature_loss
            loss.backward()
            
            instance_epoch_loss += instance_loss.item()
            feature_epoch_loss += feature_loss.item()
            epoch_loss += loss.item()

            if config["use_gradnorm"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["max_norm"])
            optimizer.step()

            wandb.log({
                "instance_batch_loss": instance_loss.item(),
                "feature_batch_loss": feature_loss.item(),
                "batch_loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]['lr'],
            })

            progress_bar.set_postfix(batch_loss=epoch_loss / (i + 1))
        
        if config["use_scheduler"]:
            scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        avg_instance_loss = instance_epoch_loss / len(train_loader)
        avg_feature_loss = feature_epoch_loss / len(train_loader)

        wandb.log({
            "instance_epoch_loss": avg_instance_loss,
            "feature_epoch_loss": avg_feature_loss,
            "epoch_loss": avg_loss,
            "epoch": epoch
        })
        print(f'Epoch [{epoch}], Loss: {avg_loss}, Instance Loss: {avg_instance_loss}, Feature Loss: {avg_feature_loss}')

        nmi_backbone, ari_backbone, acc_backbone, nmi_feature, ari_feature, acc_feature = evaluate(model, test_loader, device)
        print(f'Backbone NMI: {nmi_backbone}, Feature NMI: {nmi_feature}')
        wandb.log({
            "NMI_backbone": nmi_backbone,
            "ARI_backbone": ari_backbone,
            "ACC_backbone": acc_backbone,
            "NMI_feature": nmi_feature,
            "ARI_feature": ari_feature,
            "ACC_feature": acc_feature,
        })
            

        best_nmi = save_best_model(nmi_backbone, nmi_feature, best_nmi, config, model, wandb.run.name)
        records.update_best_metrics(nmi_backbone, ari_backbone, acc_backbone, nmi_feature, ari_feature, acc_feature)
        

        if (epoch) % 10 == 0 and config["class_num"] <= 20:
            visualize_embeddings(model, visualize_loader, device, epoch, wandb.run.name)

        if (epoch) % 100 == 0:
            current_metrics = {
                "nmi_backbone": nmi_backbone,
                "ari_backbone": ari_backbone,
                "acc_backbone": acc_backbone,
                "nmi_feature": nmi_feature,
                "ari_feature": ari_feature,
                "acc_feature": acc_feature,
            }
            records.log_best_metrics(epoch)
            records.log_current_metrics(current_metrics)

        if config["checkpoint"]:
            save_checkpoint(model, optimizer, epoch, wandb.run.id, records, best_nmi, scheduler, filename=checkpoint_path)

    records.save_csv(wandb.run.name)
    move_model_to_finished(config, wandb.run.name)
    
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run experiment with specified dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., cifar10, cifar100, imagenet10)')
    args = parser.parse_args()
    
    config_path = 'config/config.yaml'
    config = load_config(config_path, args.dataset)
    config["dataset"] = args.dataset
    
    run(config)
    