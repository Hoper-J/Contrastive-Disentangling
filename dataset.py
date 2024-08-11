import os
import sys
import pickle
import numpy as np
import cv2
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from torchvision import datasets, transforms


class AugmentedDataset(Dataset):
    def __init__(self, dataset, augmentation_transform):
        self.dataset = dataset
        self.augmentation_transform = augmentation_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        x1 = self.augmentation_transform(x)
        x2 = self.augmentation_transform(x)
        return x1, x2, y


class BaseTransformDataset(Dataset):
    def __init__(self, dataset, base_transform):
        self.dataset = dataset
        self.base_transform = base_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        x = self.base_transform(x)
        return x, y


class GaussianBlur:
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)
        prob = np.random.random_sample()
        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)
        return sample


def get_transforms(s, blur):
    """
    Get base and augmentation transforms based on the parameters.
    """
    base_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor()
    ])

    augmentation_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        GaussianBlur(kernel_size=23) if blur else transforms.Lambda(lambda x: x),
        transforms.ToTensor()
    ])

    return base_transform, augmentation_transform


def load_dataset(dataset_name, root='./data', download=True, transform=None):
    """
    Load dataset based on the dataset name and return train and test datasets.
    """
    dataset_mapping = {
        'cifar10': lambda: (
            datasets.CIFAR10(root=root, train=True, download=download, transform=transform),
            datasets.CIFAR10(root=root, train=False, download=download, transform=transform)
        ),
        'cifar100': lambda: (
            CIFAR100(root=root, train=True, download=download, transform=transform),
            CIFAR100(root=root, train=False, download=download, transform=transform)
        ),
        'imagenet10': lambda: (
            datasets.ImageFolder(root='data/imagenet-10', transform=transform),
            datasets.ImageFolder(root='data/imagenet-10', transform=transform)
        ),
        'stl10': lambda: (
            datasets.STL10(root=root, split='train', download=download, transform=transform),
            datasets.STL10(root=root, split='test', download=download, transform=transform)
        ),
        'tiny-imagenet': lambda: (
            datasets.ImageFolder(root='data/tiny-imagenet-200/train', transform=transform),
            datasets.ImageFolder(root='data/tiny-imagenet-200/train', transform=transform)
        ),
    }

    if dataset_name not in dataset_mapping:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return dataset_mapping[dataset_name]()


def get_combined_datasets():
    """
    Combine multiple datasets into a single dataset.
    """
    cifar10_train, cifar10_test = load_dataset('cifar10')
    cifar100_train, cifar100_test = load_dataset('cifar100')
    imagenet10, _ = load_dataset('imagenet10')
    stl10_train, stl10_test = load_dataset('stl10')

    combined_dataset = ConcatDataset([
        cifar10_train, cifar10_test,
        cifar100_train, cifar100_test,
        imagenet10,
        stl10_train, stl10_test
    ])

    return combined_dataset


def get_data_loader(config):
    """
    Get train, test, and visualization data loaders based on the config.
    """
    base_transform, augmentation_transform = get_transforms(config['s'], config['blur'])

    if config['use_combined_datasets']:
        train_dataset = get_combined_datasets()
        _, test_dataset = load_dataset(config["dataset"])
    else:
        train_dataset, test_dataset = load_dataset(config["dataset"])

        # Concatenate train and test datasets
        if config['dataset'] not in ['imagenet10', 'tiny-imagenet']:
            dataset = ConcatDataset([train_dataset, test_dataset])
            train_dataset = dataset
            test_dataset = dataset

    # Subset for visualization
    visualize_indices = np.random.choice(len(test_dataset), 1000, replace=False)
    visualize_data = Subset(test_dataset, visualize_indices)

    augmented_train_dataset = AugmentedDataset(train_dataset, augmentation_transform)
    transformed_vis_dataset = BaseTransformDataset(visualize_data, base_transform)
    transformed_test_dataset = BaseTransformDataset(test_dataset, base_transform)

    train_loader = DataLoader(augmented_train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(transformed_test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4, drop_last=True)
    visualize_loader = DataLoader(transformed_vis_dataset, batch_size=config["batch_size"], shuffle=False)

    return train_loader, test_loader, visualize_loader


class CIFAR100(datasets.CIFAR10):
    """CIFAR100 Dataset with 20 super-classes."""
    
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    
    meta = {
        'filename': 'meta',
        'key': 'coarse_label_names',  # Change this to coarse_label_names
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }

    def __init__(self, *args, **kwargs):
        super(CIFAR100, self).__init__(*args, **kwargs)
        
        # Determine which list to use based on train or test mode
        downloaded_list = self.train_list if self.train else self.test_list
        
        # Initialize list to hold coarse labels
        self.coarse_labels = []
        
        # Load coarse labels from the dataset
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.coarse_labels.extend(entry['coarse_labels'])
        
        # Use 20 super-classes as ground-truth
        self.targets = self.coarse_labels

        # Load coarse class names
        self._load_meta()

    def _load_meta(self):
        """Load the CIFAR-100 metadata."""
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not os.path.isfile(path):
            raise RuntimeError('Metadata file not found or corrupted.')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        
        # Check if classes loaded correctly
        if len(self.classes) != 20:
            raise RuntimeError(f'Expected 20 super-classes, but got {len(self.classes)}.')
