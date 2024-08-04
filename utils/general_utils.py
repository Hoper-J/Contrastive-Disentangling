import os
import shutil
import torch
import yaml
import numpy as np

    
def load_config(config_path, dataset):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        common_config = config['common']
        dataset_config = config['datasets'][dataset]
        common_config.update(dataset_config)
        return common_config
        
def set_seed(seed):
    """
    Set the seed for random number generators to ensure reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_experiment_name(config, remark=""):
    """
    Generate a name for the experiment based on the configuration.

    Parameters:
    - config (dict): Configuration dictionary containing experiment settings.
    - remark (str, optional): Additional remark or note to append to the experiment name.

    Returns:
    - str: Experiment name.
    """
    parts = [
        f"bs{config['batch_size']}",
        # f"ep{config['epochs']}",
        f"lr{config['learning_rate']}",
        config["backbone"],
    ]

    if config["use_scheduler"]:
        parts.append("scheduler")
    
    if not config["use_gradnorm"]:
        parts.append("nogradnorm")

    if config["weight_decay"] != 0:
        parts.append(f"wd{config['weight_decay']}")

    # Append the remark if provided
    if remark:
        parts.append(remark)
        
    return "_".join(parts)

    
def count_parameters(model):
    """
    Calculate the number of trainable parameters in a model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_best_model(nmi_backbone, nmi_classifier, best_nmi, config, model, model_name):
    """
    Save the model if it achieves the best NMI score so far.

    Parameters:
    - nmi_backbone (float): NMI score for the backbone features.
    - nmi_classifier (float): NMI score for the classifier outputs.
    - best_nmi (float): Best NMI score so far.
    - config (dict): Configuration dictionary.
    - model: The model to save.
    - model_name (str): Name of the saved model.

    Returns:
    - float: Updated best NMI score.
    """
    if nmi_backbone > best_nmi or nmi_classifier > best_nmi:
        best_nmi = max(nmi_backbone, nmi_classifier)
        if config["save_model"]:
            model_path = os.path.join('models', f'best_model_{model_name}.pth')
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model with NMI {best_nmi:.4f} to {model_path}")
        else:
            print(f"Best model's NMI {best_nmi:.4f}")
    return best_nmi

def move_model_to_finished(cfg, model_name):
    """
    Move the saved model to the 'finished' folder.

    Parameters:
    - cfg: Configuration dictionary.
    - model_name (str): Name of the saved model.
    """
    if cfg["save_model"]:
        best_model_path = os.path.join('models', f'best_model_{model_name}.pth')
        finished_model_path = os.path.join('models', 'finished', os.path.basename(best_model_path))
        print(best_model_path, finished_model_path)
        os.makedirs(os.path.dirname(finished_model_path), exist_ok=True)
        shutil.move(best_model_path, finished_model_path)
        print(f"Moved best model to {finished_model_path}")


