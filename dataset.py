import torch
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
from torchvision import datasets, transforms
import numpy as np
from torchvision.transforms import TrivialAugmentWide
import cv2
from PIL import Image

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
    
def get_data_loader(dataset_name, batch_size, s=1.0,blur = True):
    dataset_mapping = {
        'mnist': datasets.MNIST,
        'fashion-mnist': datasets.FashionMNIST,
        'cifar10': datasets.CIFAR10,
        'imagenet10': datasets.ImageFolder
    }

    if dataset_name not in dataset_mapping:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if dataset_name == 'cifar10':
        s=0.5
        blur = False
    dataset_class = dataset_mapping[dataset_name]

    base_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)) if dataset_name in ['mnist', 'fashion-mnist'] else lambda x:x        
    ])

    augmentation_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)],
                                p=0.8),
        transforms.RandomGrayscale(p=0.2),
        
        GaussianBlur(kernel_size=23) if blur else lambda x: x,
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)) if dataset_name in ['mnist', 'fashion-mnist'] else lambda x:x
    ])
    
    if dataset_name == 'imagenet10':
        train_dataset = dataset_class(root='data/imagenet-10', transform=None)
        test_dataset = dataset_class(root='data/imagenet-10', transform=None)
        
        
    else:
        train_dataset = dataset_class(root='./data', train=True, download=True, transform=None)
        test_dataset = dataset_class(root='./data', train=False, download=True, transform=None)

        dataset = ConcatDataset([train_dataset, test_dataset])
        train_dataset = dataset
        test_dataset = dataset

    visualize_indices = np.random.choice(len(test_dataset), 1000, replace=False)
    visualize_data = torch.utils.data.Subset(test_dataset, visualize_indices)

    augmented_train_dataset = AugmentedDataset(train_dataset, augmentation_transform)
    transformed_vis_dataset = BaseTransformDataset(visualize_data, base_transform)
    transformed_test_dataset = BaseTransformDataset(test_dataset, base_transform)
    
    train_loader = DataLoader(augmented_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(transformed_test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)
    visualize_loader = DataLoader(transformed_vis_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, visualize_loader
