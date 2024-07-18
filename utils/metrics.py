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
    - nmi_classifier: NMI score for the classifier outputs.
    - ari_classifier: ARI score for the classifier outputs.
    - acc_classifier: ACC score for the classifier outputs.
    """
    model.eval()
    embeddings_backbone = []
    embeddings_classifier = []
    labels = []
    with torch.no_grad():
        for (x, y) in loader:
            x = x.to(device)
            h = model.resnet(x)
            c = model.forward_cluster(x)
            embeddings_backbone.append(h.cpu().numpy())
            embeddings_classifier.append(c.cpu().numpy())
            labels.append(y.numpy())
    embeddings_backbone = np.concatenate(embeddings_backbone, axis=0)
    embeddings_classifier = np.concatenate(embeddings_classifier, axis=0)
    labels = np.concatenate(labels, axis=0)

    # 使用 K-means 聚类
    n_clusters = len(np.unique(labels))
    kmeans_backbone = KMeans(n_clusters=n_clusters, random_state=42).fit(embeddings_backbone)

    # 计算 NMI 和 ARI
    nmi_backbone = metrics.normalized_mutual_info_score(labels, kmeans_backbone.labels_)
    ari_backbone = metrics.adjusted_rand_score(labels, kmeans_backbone.labels_)

    nmi_classifier = metrics.normalized_mutual_info_score(labels, embeddings_classifier)
    ari_classifier = metrics.adjusted_rand_score(labels, embeddings_classifier)

    # 计算 ACC
    acc_backbone = calculate_accuracy(labels, kmeans_backbone.labels_, n_clusters)
    acc_classifier = calculate_accuracy(labels, embeddings_classifier, n_clusters)

    return nmi_backbone, ari_backbone, acc_backbone, nmi_classifier, ari_classifier, acc_classifier

def calculate_accuracy(true_labels, cluster_labels, num_classes):
    """
    Calculate the clustering accuracy using the Hungarian algorithm.

    Parameters:
    - true_labels: Ground truth labels.
    - cluster_labels: Cluster labels predicted by the model.
    - num_classes: Number of unique classes.

    Returns:
    - accuracy: Clustering accuracy.
    """
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion_matrix, (true_labels, cluster_labels), 1)

    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
    accuracy = confusion_matrix[row_ind, col_ind].sum() / len(true_labels)
    return accuracy
