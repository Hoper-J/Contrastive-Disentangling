import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import wandb
import torch


def visualize_embeddings(model, loader, device, epoch, name):
    """
    Visualize embeddings using t-SNE and log the visualization to wandb.

    Parameters:
    - model: The model to extract embeddings from.
    - loader: DataLoader for the dataset.
    - device: The device to run the model on (CPU/GPU).
    - epoch (int): The current epoch number.
    - name (str): The name to use for the saved visualization image.
    """
    model.eval()
    embeddings_backbone = []
    embeddings_feature = []
    labels = []
    with torch.no_grad():
        for (x, y) in loader:
            x = x.to(device)
            h, f = model.extract_backbone_and_feature(x)
            embeddings_backbone.append(h.cpu().numpy())
            embeddings_feature.append(f.cpu().numpy())
            labels.append(y.numpy())
    embeddings_backbone = np.concatenate(embeddings_backbone, axis=0)
    embeddings_feature = np.concatenate(embeddings_feature, axis=0)
    labels = np.concatenate(labels, axis=0)

    # TSNE for embeddings_backbone
    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(embeddings_backbone)

    # Choose colormap based on number of classes
    num_classes = len(np.unique(labels))
    colormap = get_colormap(num_classes)

    # Plot t-SNE for backbone embeddings
    tsne_results = TSNE(n_components=2, random_state=0).fit_transform(embeddings_backbone)
    plot_tsne(tsne_results, labels, colormap, f'Epoch {epoch} ({name})', 'backbone', epoch, name)

    # Plot t-SNE for feature embeddings
    tsne_feature_results = TSNE(n_components=2, random_state=0).fit_transform(embeddings_feature)
    plot_tsne(tsne_feature_results, labels, colormap, f'Epoch {epoch} ({name})', 'feature', epoch, name)

    wandb.log({
        "TSNE_backbone": wandb.Image(f'images/tsne_embeddings_backbone_epoch_{epoch}_{name}.png'),
        "TSNE_feature": wandb.Image(f'images/tsne_embeddings_feature_epoch_{epoch}_{name}.png'),
        "epoch": epoch
    })

def get_colormap(num_classes):
    """Return appropriate colormap based on number of classes."""
    if num_classes <= 10:
        return 'tab10'
    elif num_classes <= 20:
        return 'tab20'
    else:
        return 'viridis'  # or any other suitable colormap

def plot_tsne(tsne_results, labels, colormap, title, embedding_type, epoch, name):
    """Helper function to plot t-SNE results."""
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap=colormap, s=1)
    plt.colorbar(scatter)
    plt.title(title)
    tsne_path = os.path.join('images', f'tsne_embeddings_{embedding_type}_epoch_{epoch}_{name}.png')
    os.makedirs(os.path.dirname(tsne_path), exist_ok=True)
    plt.savefig(tsne_path)
    plt.close()
