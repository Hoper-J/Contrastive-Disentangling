import wandb


DEFAULT_METRICS = [
    "epoch_loss", "instance_epoch_loss", "cluster_epoch_loss",
    "NMI_backbone", "ARI_backbone", "NMI_feature", "ARI_feature", 
    "ACC_backbone", "ACC_feature", "TSNE_backbone", "TSNE_feature"
]

def init_wandb(config, experiment_name, run_id=None, metrics=DEFAULT_METRICS):
    """
    Initializes Weights & Biases (wandb) for logging experiments.

    Parameters:
    - config (dict): Configuration dictionary containing experiment settings.
    - experiment_name (str): Name of the experiment.
    - run_id (str, optional): ID of the run to resume from. Defaults to None.
    - metrics (list, optional): List of metrics to log. Defaults to DEFAULT_METRICS.
    """
    if run_id is None:
        wandb.init(project=config["project"], name=experiment_name, config=config, reinit=True)
        wandb.save("train.py")
        wandb.save("loss.py")
        wandb.save("dataset.py")
        wandb.save("modules/network.py")
        wandb.save("config/config.yaml")
    else:
        wandb.init(project=config["project"], name=experiment_name, id=run_id, resume="must", config=config, reinit=True)

    for metric in metrics:
        wandb.define_metric(metric, step_metric="epoch")


def log_batch_metrics(instance_batch_loss, feature_batch_loss, batch_loss, learning_rate):
    """
    Logs the metrics for each batch during training to Weights & Biases (wandb).

    Parameters:
    - instance_batch_loss (float): The instance loss for the current batch.
    - feature_batch_loss (float): The feature loss for the current batch.
    - batch_loss (float): The total loss for the current batch.
    - learning_rate (float): The learning rate used for training.
    """
    wandb.log({
        "instance_batch_loss": instance_batch_loss,
        "feature_batch_loss": feature_batch_loss,
        "batch_loss": batch_loss,
        "learning_rate": learning_rate,
    })
    
def log_epoch_metrics(epoch, avg_instance_loss, avg_feature_loss, avg_loss, nmi_backbone, ari_backbone, acc_backbone, nmi_feature, ari_feature, acc_feature):
    """
    Logs the metrics for the current epoch to Weights & Biases (wandb).

    Parameters:
    - epoch (int): The current epoch number.
    - avg_instance_loss (float): Average instance loss for the epoch.
    - avg_feature_loss (float): Average feature loss for the epoch.
    - avg_loss (float): Average total loss for the epoch.
    - nmi_backbone (float): Normalized Mutual Information (NMI) for the backbone features.
    - ari_backbone (float): Adjusted Rand Index (ARI) for the backbone features.
    - acc_backbone (float): Accuracy for the backbone features.
    - nmi_feature (float): Normalized Mutual Information (NMI) for the feature predictions.
    - ari_feature (float): Adjusted Rand Index (ARI) for the feature predictions.
    - acc_feature (float): Accuracy for the feature predictions.
    """
    wandb.log({
        "epoch": epoch,
        "instance_epoch_loss": avg_instance_loss,
        "feature_epoch_loss": avg_feature_loss,
        "epoch_loss": avg_loss,
        "NMI_backbone": nmi_backbone,
        "ARI_backbone": ari_backbone,
        "ACC_backbone": acc_backbone,
        "NMI_feature": nmi_feature,
        "ARI_feature": ari_feature,
        "ACC_feature": acc_feature
    })

def log_tsne_images(epoch, run_name):
    """
    Logs t-SNE images for the current epoch to Weights & Biases (wandb).

    Parameters:
    - epoch (int): The current epoch number, used for labeling the images in the logs.
    - run_name (str): The name of the current wandb run, used to identify the specific images.

    Images are expected to be stored in the 'images' directory with filenames following 
    the pattern 'tsne_embeddings_backbone_epoch_<epoch>_<run_name>.png' and 
    'tsne_embeddings_feature_epoch_<epoch>_<run_name>.png'.
    """
    wandb.log({
        "TSNE_backbone": wandb.Image(f"images/tsne_embeddings_backbone_epoch_{epoch}_{run_name}.png"),
        "TSNE_feature": wandb.Image(f"images/tsne_embeddings_feature_epoch_{epoch}_{run_name}.png"),
    })