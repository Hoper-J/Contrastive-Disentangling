import os
import argparse

import torch

from dataset import get_data_loader
from modules.network import Network
from utils.general_utils import load_config, set_seed, get_experiment_name
from utils.metrics import evaluate


def evaluate_model(dataset, config_path='config/config.yaml', model_path=None, epoch=None):
    """
    Evaluate the performance of a pre-trained model on a specified dataset.
    
    Parameters:
    - dataset (str): Name of the dataset (e.g., 'cifar10', 'cifar100', 'imagenet10', 'stl10').
    - config_path (str): Path to the configuration file (default: 'config/config.yaml').
    - model_path (str, optional): Path to the saved model file. If not provided, it will be generated automatically.
    - epoch (int, optional): The epoch number to evaluate. Defaults to the final epoch if not provided. Ignored if model_path is specified.
    
    Returns:
    - results (dict): Evaluation metrics for the model.
    """
    
    # Load the configuration
    config = load_config(config_path, dataset)
    config["dataset"] = dataset

    set_seed(config["seed"])
    
    # Determine experiment name and model name
    experiment_name = get_experiment_name(config)

    # If epoch is not provided, use the final epoch in the config
    if epoch is None:
        epoch = config["epochs"]

    # If model_path is not provided, generate the path based on the epoch
    if model_path is None:
        model_dir = os.path.join("models", config["dataset"])
        model_path = os.path.join(model_dir, f"model_epoch_{epoch}.pth")

        # Check if the model path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"The model for epoch {epoch} could not be found at {model_path}.")
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = Network(config["backbone"], config["feature_num"], config["hidden_dim"]).to(device)
    
    # Load the saved state_dict from the model_path
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    
    # Set the model to evaluation mode
    model.eval()

    # Load the test dataset
    _, test_loader, _ = get_data_loader(config)

    # Evaluate the model
    nmi_backbone, ari_backbone, acc_backbone, nmi_feature, ari_feature, acc_feature = evaluate(model, test_loader, device)
    
    # Return the evaluation metrics as a dictionary
    results = {
        "NMI_backbone": nmi_backbone,
        "ARI_backbone": ari_backbone,
        "ACC_backbone": acc_backbone,
        "NMI_feature": nmi_feature,
        "ARI_feature": ari_feature,
        "ACC_feature": acc_feature
    }
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a pre-trained model on a specified dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., cifar10, cifar100, stl10, imagenet10)")
    parser.add_argument("--model", type=str, help="Path to the saved model file (optional)")
    parser.add_argument("--epoch", type=int, help="Epoch number to evaluate (optional). Defaults to the last epoch in the config.")
    args = parser.parse_args()

    # Evaluate the model
    results = evaluate_model(dataset=args.dataset, model_path=args.model, epoch=args.epoch)
    
    # Print the results
    print("Evaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value}")
