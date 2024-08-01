import numpy as np
import torch
from sklearn import metrics
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

def evaluate(model, loader, device):
    """
    Evaluate the model on a dataset.

    Parameters:
    - model: The model to evaluate.
    - loader: DataLoader for the dataset.
    - device: The device to run the model on (CPU/GPU).

    Returns:
    - nmi_backbone: NMI score for the backbone features.
    - ari_backbone: ARI score for the backbone features.
    - acc_backbone: ACC score for the backbone features.
    - nmi_feature: NMI score for the feature outputs.
    - ari_feature: ARI score for the feature outputs.
    - acc_feature: ACC score for the feature outputs.
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

    # Concatenate all the extracted features and labels
    embeddings_backbone = np.concatenate(embeddings_backbone, axis=0)
    embeddings_feature = np.concatenate(embeddings_feature, axis=0)
    labels = np.concatenate(labels, axis=0)

    # Use K-means clustering to obtain cluster labels for both backbone and feature outputs
    n_clusters = len(np.unique(labels))
    kmeans_backbone = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(embeddings_backbone)
    kmeans_feature = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(embeddings_feature)
    
    # Calculate NMI and ARI for backbone features
    nmi_backbone = metrics.normalized_mutual_info_score(labels, kmeans_backbone.labels_)
    ari_backbone = metrics.adjusted_rand_score(labels, kmeans_backbone.labels_)

    # Calculate NMI and ARI for feature head outputs
    nmi_feature = metrics.normalized_mutual_info_score(labels, kmeans_feature.labels_)
    ari_feature = metrics.adjusted_rand_score(labels, kmeans_feature.labels_)

    # Calculate ACC for backbone features
    acc_backbone = calculate_accuracy(labels, kmeans_backbone.labels_, n_clusters)

    # Calculate ACC for feature head outputs
    acc_feature = calculate_accuracy(labels, kmeans_feature.labels_, n_clusters)

    # Return the calculated metrics
    return nmi_backbone, ari_backbone, acc_backbone, nmi_feature, ari_feature, acc_feature


def calculate_accuracy(true_labels, cluster_labels, num_classes):
    """
    Calculate the clustering accuracy using the Hungarian algorithm.

    Parameters:
    - true_labels: Ground truth labels.
    - cluster_labels: Cluster labels obtained from K-means clustering.
    - num_classes: Number of unique classes.

    Returns:
    - accuracy: Clustering accuracy.
    """
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion_matrix, (true_labels, cluster_labels), 1)

    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
    accuracy = confusion_matrix[row_ind, col_ind].sum() / len(true_labels)
    return accuracy
