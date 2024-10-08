import os
import pandas as pd


class ExperimentRecords:
    def __init__(self):
        self.records = {
            "epoch": [],
            "best_nmi_feature": [],
            "best_ari_feature": [],
            "best_acc_feature": [],
            "best_nmi_backbone": [],
            "best_ari_backbone": [],
            "best_acc_backbone": [],
            "nmi_backbone": [],
            "ari_backbone": [],
            "acc_backbone": [],
            "nmi_feature": [],
            "ari_feature": [],
            "acc_feature": [],
        }
        self.best_metrics = {
            "nmi_feature": 0.0,
            "ari_feature": 0.0,
            "acc_feature": 0.0,
            "nmi_backbone": 0.0,
            "ari_backbone": 0.0,
            "acc_backbone": 0.0,
        }

    def update_best_metrics(self, nmi_backbone: float, ari_backbone: float, acc_backbone: float, nmi_feature: float, ari_feature: float, acc_feature: float):
        self.best_metrics["nmi_feature"] = max(self.best_metrics["nmi_feature"], nmi_feature)
        self.best_metrics["ari_feature"] = max(self.best_metrics["ari_feature"], ari_feature)
        self.best_metrics["acc_feature"] = max(self.best_metrics["acc_feature"], acc_feature)
        self.best_metrics["nmi_backbone"] = max(self.best_metrics["nmi_backbone"], nmi_backbone)
        self.best_metrics["ari_backbone"] = max(self.best_metrics["ari_backbone"], ari_backbone)
        self.best_metrics["acc_backbone"] = max(self.best_metrics["acc_backbone"], acc_backbone)

    def log_best_metrics(self, epoch: int):
        self.records["epoch"].append(epoch)
        for key, value in self.best_metrics.items():
            self.records[f"best_{key}"].append(value)

    def log_current_metrics(self, nmi_backbone: float, ari_backbone: float, acc_backbone: float, nmi_feature: float, ari_feature: float, acc_feature: float):
        self.records["nmi_feature"].append(nmi_feature)
        self.records["ari_feature"].append(ari_feature)
        self.records["acc_feature"].append(acc_feature)
        self.records["nmi_backbone"].append(nmi_backbone)
        self.records["ari_backbone"].append(ari_backbone)
        self.records["acc_backbone"].append(acc_backbone)

    def save_csv(self, run_name: str):
        df = pd.DataFrame(self.records).T
        os.makedirs('csv', exist_ok=True)
        df.to_csv(os.path.join('csv', f'records_{run_name}.csv'), index=True)
