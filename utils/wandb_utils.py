import wandb

DEFAULT_METRICS = [
    "epoch_loss", "instance_epoch_loss", "cluster_epoch_loss",
    "NMI_backbone", "ARI_backbone", "NMI_classifier", "ARI_classifier", 
    "ACC_backbone", "ACC_classifier", "TSNE_backbone", "TSNE_classifier"
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
        wandb.save("modules/network.py")
    else:
        wandb.init(project=config["project"], name=experiment_name, id=run_id, resume="must", config=config, reinit=True)

    for metric in metrics:
        wandb.define_metric(metric, step_metric="epoch")