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
from loss import InstanceLoss, VarientialClusterLoss
from modules.network import Network
from utils.general_utils import set_seed, count_parameters, save_best_model, move_model_to_finished
from utils.wandb_utils import init_wandb
from utils.metrics import evaluate
from utils.visualization import visualize_embeddings
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.experiment_records import ExperimentRecords


def get_experiment_name(config):
    """
    Generate a name for the experiment based on the configuration.

    Parameters:
    - config (dict): Configuration dictionary containing experiment settings.

    Returns:
    - str: Experiment name.
    """
    parts = [
        f"bs{config['batch_size']}",
        # f"ep{config['epochs']}",
        f"lr{config['learning_rate']}",
        config["backbone"],
        "inst" if config["instance"] else "no inst",
        "cluster" if config["cluster"] else "nocluster",
        "variational" if config["variational"] else "novariational", "weight1"
    ]
    return "_".join(parts)
    

def run(config):
    set_seed(config["seed"])
    experiment_name = get_experiment_name(config)
    config["project"] = f"{config['project']}-{config['dataset']}"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    train_loader, test_loader, visualize_loader = get_data_loader(config["dataset"], config["batch_size"])
    model = Network(config["backbone"], 10, config["batch_size"], 128, config["variational"]).to(device)
    print(f'The model has {count_parameters(model):,} trainable parameters.')

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=0)

    instance_loss_fn = InstanceLoss(config["batch_size"], 0.5, device=device)
    cluster_loss_fn = VarientialClusterLoss(10, 1, device=device, use_variational=config["variational"], var_weight=1)

    records = ExperimentRecords()
    best_nmi = 0.0
    
    start_epoch = 1
    run_id = None
    checkpoint_path = f'checkpoints/{config["dataset"]}_checkpoint_{experiment_name}.pth.tar'
    
    if config["reload"] and os.path.exists(checkpoint_path):
        start_epoch, run_id, best_nmi = load_checkpoint(model, optimizer, records, filename=checkpoint_path)
        
    init_wandb(config, experiment_name, run_id)
        
    for epoch in range(start_epoch, config["epochs"]+1):
        model.train()
        instance_epoch_loss = 0.0
        cluster_epoch_loss = 0.0
        variational_epoch_loss = 0.0
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for i, (x_i, x_j, _) in enumerate(progress_bar):
            x_i, x_j = x_i.to(device), x_j.to(device)
            
            optimizer.zero_grad()

            z_i, z_j, out_i, out_j, entropy_loss = model(x_i, x_j)

            instance_loss = instance_loss_fn(z_i, z_j)
            cluster_loss, variational_loss = cluster_loss_fn(out_i, out_j)
                
            loss = instance_loss + cluster_loss + variational_loss #+ entropy_loss
            loss.backward()
            
            instance_epoch_loss += instance_loss.item()
            cluster_epoch_loss += cluster_loss.item()
            variational_epoch_loss += variational_loss.item()
            epoch_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            wandb.log({
                "instance_batch_loss": instance_loss.item(),
                "cluster_batch_loss": cluster_loss.item(),
                "var_batch_loss": variational_loss.item(),
                "batch_loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]['lr'],
            })

            progress_bar.set_postfix(batch_loss=epoch_loss / (i + 1))
                
        avg_loss = epoch_loss / len(train_loader)
        avg_instance_loss = instance_epoch_loss / len(train_loader)
        avg_cluster_loss = cluster_epoch_loss / len(train_loader)
        avg_variational_loss = variational_epoch_loss / len(train_loader)

        wandb.log({
            "instance_epoch_loss": avg_instance_loss,
            "cluster_epoch_loss": avg_cluster_loss,
            "var_epoch_loss": avg_variational_loss,
            "epoch_loss": avg_loss,
            "epoch": epoch
        })
        print(f'Epoch [{epoch}], Loss: {avg_loss}, Instance Loss: {avg_instance_loss}, Cluster Loss: {avg_cluster_loss}, Variational Loss: {avg_variational_loss}')

        nmi_backbone, ari_backbone, acc_backbone, nmi_classifier, ari_classifier, acc_classifier = evaluate(model, test_loader, device)
        print(f'Backbone NMI: {nmi_backbone}, Classifier NMI: {nmi_classifier}')
        wandb.log({
            "NMI_backbone": nmi_backbone,
            "ARI_backbone": ari_backbone,
            "ACC_backbone": acc_backbone,
            "NMI_classifier": nmi_classifier,
            "ARI_classifier": ari_classifier,
            "ACC_classifier": acc_classifier,
        })
            

        best_nmi = save_best_model(nmi_backbone, nmi_classifier, best_nmi, config, model)
        records.update_best_metrics(nmi_backbone, ari_backbone, acc_backbone, nmi_classifier, ari_classifier, acc_classifier)
        

        if (epoch) % 10 == 0:
            visualize_embeddings(model, visualize_loader, device, epoch, wandb.run.name)

        if (epoch) % 100 == 0:
            current_metrics = {
                "nmi_backbone": nmi_backbone,
                "ari_backbone": ari_backbone,
                "acc_backbone": acc_backbone,
                "nmi_classifier": nmi_classifier,
                "ari_classifier": ari_classifier,
                "acc_classifier": acc_classifier,
            }
            records.log_best_metrics(epoch)
            records.log_current_metrics(current_metrics)

        if config["checkpoint"]:
            save_checkpoint(model, optimizer, epoch, wandb.run.id, records, best_nmi, filename=checkpoint_path)

    records.save_csv(wandb.run.name)
    move_model_to_finished(config, os.path.join('models', f'best_model_{wandb.run.name}.pth'))
    
    wandb.finish()


config = {
        "dataset": "imagenet10",
        "batch_size": 128,
        "epochs": 600,
        "learning_rate": 3e-4,
        "backbone": "ResNet34",
        "instance": True,
        "cluster": True,
        "variational": True,
        "checkpoint": True,
        "reload": True,
        "save_model": True,
        "project": "CVC",
        "seed": 42,
    }
    
run(config)