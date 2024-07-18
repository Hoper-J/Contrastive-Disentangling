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
    embeddings_classifier = []
    labels = []
    with torch.no_grad():
        for (x, y) in loader:
            x = x.to(device)
            h = model.resnet(x)
            if model.use_variational:
                c = model.class_projector(h).logit_predictive.loc
            else:
                c = model.class_projector(h)
            embeddings_backbone.append(h.cpu().numpy())
            embeddings_classifier.append(c.cpu().numpy())
            labels.append(y.numpy())
    embeddings_backbone = np.concatenate(embeddings_backbone, axis=0)
    embeddings_classifier = np.concatenate(embeddings_classifier, axis=0)
    labels = np.concatenate(labels, axis=0)

    # TSNE for embeddings_backbone
    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(embeddings_backbone)

    plt.figure(figsize=(8, 8))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='tab10', s=1)
    plt.colorbar()
    plt.title(f'Epoch {epoch} ({name})')
    tsne_backbone_path = os.path.join('images', f'tsne_embeddings_backbone_epoch_{epoch}_{name}.png')
    os.makedirs(os.path.dirname(tsne_backbone_path), exist_ok=True)
    plt.savefig(tsne_backbone_path)
    plt.close()

    # TSNE for embeddings_classifier
    tsne_classifier = TSNE(n_components=2, random_state=0)
    tsne_classifier_results = tsne_classifier.fit_transform(embeddings_classifier)

    plt.figure(figsize=(8, 8))
    plt.scatter(tsne_classifier_results[:, 0], tsne_classifier_results[:, 1], c=labels, cmap='tab10', s=1)
    plt.colorbar()
    plt.title(f'Epoch {epoch} ({name})')
    tsne_classifier_path = os.path.join('images', f'tsne_embeddings_classifier_epoch_{epoch}_{name}.png')
    os.makedirs(os.path.dirname(tsne_classifier_path), exist_ok=True)
    plt.savefig(tsne_classifier_path)
    plt.close()

    wandb.log({
        "TSNE_backbone": wandb.Image(tsne_backbone_path),
        "TSNE_classifier": wandb.Image(tsne_classifier_path),
        "epoch": epoch
    })
