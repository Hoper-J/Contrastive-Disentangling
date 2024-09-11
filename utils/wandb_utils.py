import wandb


class WandBLogger:
    DEFAULT_METRICS = [
        "epoch_loss", "instance_epoch_loss", "cluster_epoch_loss",
        "NMI_backbone", "ARI_backbone", "NMI_feature", "ARI_feature", 
        "ACC_backbone", "ACC_feature", "TSNE_backbone", "TSNE_feature"
    ]
    
    def __init__(self, config, experiment_name=None, run_id=None):
        self.use_wandb = config.get("use_wandb", False)
        if self.use_wandb:
            self._init_wandb(config, experiment_name, run_id)
            self.run_name = wandb.run.name
            self.run_id = wandb.run.id
        else:
            self.run_name = None
            self.run_id = None

    def _init_wandb(self, config, experiment_name, run_id=None, metrics=DEFAULT_METRICS):
        """
        Initializes Weights & Biases (wandb) for logging experiments.
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

    def log_batch_metrics(self, instance_loss, feature_loss, loss, lr):
        """
        Logs the metrics for each batch during training to Weights & Biases (wandb).
        """
        if self.use_wandb:
            wandb.log({
                "instance_batch_loss": instance_loss,
                "feature_batch_loss": feature_loss,
                "batch_loss": loss,
                "learning_rate": lr,
            })

    def log_epoch_metrics(self, epoch, avg_instance_loss, avg_feature_loss, avg_loss, 
                          nmi_backbone, ari_backbone, acc_backbone, nmi_feature, ari_feature, acc_feature):
        """
        Logs the metrics for the current epoch to Weights & Biases (wandb).
        """
        if self.use_wandb:
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

    def log_tsne_images(self, epoch):
        """
        Logs t-SNE images for the current epoch to Weights & Biases (wandb).
        """
        if self.use_wandb and self.run_name:
            wandb.log({
                "TSNE_backbone": wandb.Image(f"images/tsne_embeddings_backbone_epoch_{epoch}_{self.run_name}.png"),
                "TSNE_feature": wandb.Image(f"images/tsne_embeddings_feature_epoch_{epoch}_{self.run_name}.png"),
            })

    def finish(self):
        """
        Ends the wandb logging session.
        """
        if self.use_wandb:
            wandb.finish()