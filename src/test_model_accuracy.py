import gc
import random
from argparse import Namespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src import dataset
from src.model_wrappers import ModelWrapper


def test_model_accuracy(model: ModelWrapper, args: Namespace, device: torch.device, num_samples: int = 5000, random_seed: int = 42) -> None:
    """
    Test model accuracy on ImageNet test data.
    
    Args:
        model: The model to test
        args: Command line arguments containing dataset and data loading parameters
        device: The device to run the test on
        num_samples: Number of samples to test (default: 5000)
        random_seed: Random seed for sampling indices to ensure balanced class distribution (default: 42)
    """
    if args.dataset not in ['resnet_imagenet', 'regnet_imagenet', 'binary_imagenet']:
        return
    
    print("\nTesting model accuracy...")
    
    # Load ImageNet test data
    if args.dataset in ['resnet_imagenet', 'regnet_imagenet']:
        test_dataset = dataset.load_imagenet_test_data(
            test_batch_size=args.batch,
            folder=args.data_dir,
            num_workers=args.data_workers,
            pin_memory=(args.pin_memory == '1'),
            prefetch_factor=args.prefetch_factor if args.data_workers > 0 else None
        ).dataset
    else:  # binary_imagenet
        test_dataset = dataset.load_binary_imagenet_test_data(
            test_batch_size=args.batch,
            data_dir=args.data_dir,
            num_workers=args.data_workers,
            pin_memory=(args.pin_memory == '1'),
            prefetch_factor=args.prefetch_factor if args.data_workers > 0 else None
        ).dataset
    
    # Create a subset with specified number of samples using random sampling
    # to ensure balanced class distribution across all classes
    try:
        total_samples = len(test_dataset)  # type: ignore
    except (TypeError, AttributeError):
        # If dataset doesn't support len, default to num_samples
        total_samples = num_samples
    num_test_samples = min(num_samples, total_samples)
    
    # Save current random state to avoid affecting random seed in main.py
    random_state = random.getstate()
    np_random_state = np.random.get_state()
    
    # Use random sampling to ensure balanced class distribution
    random.seed(random_seed)
    np.random.seed(random_seed)
    indices = sorted(random.sample(range(total_samples), num_test_samples))
    
    # Restore random state to avoid affecting subsequent random operations in main.py
    random.setstate(random_state)
    np.random.set_state(np_random_state)
    
    test_subset = Subset(test_dataset, indices)
    
    # Create test DataLoader
    test_accuracy_loader = DataLoader(
        test_subset,
        batch_size=32,  # Use larger batch size for faster testing
        shuffle=False,
        num_workers=args.data_workers,
        pin_memory=(args.pin_memory == '1'),
        prefetch_factor=args.prefetch_factor if args.data_workers > 0 else None
    )
    
    # Test accuracy
    correct = 0
    total = 0
    model.eval()
    images = None
    labels = None
    predictions = None
    with torch.no_grad():
        for batch in test_accuracy_loader:
            if isinstance(batch, dict):
                images, labels = batch["image"], batch["label"]
            else:
                images, labels = batch
            images, labels = images.to(device), labels.to(device)
            predictions = model.predict_label(images)
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    print(f"Model accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    # Release memory
    del test_dataset, test_subset, test_accuracy_loader
    if images is not None:
        del images
    if labels is not None:
        del labels
    if predictions is not None:
        del predictions
    torch.cuda.empty_cache()
    gc.collect()
    print("Accuracy test completed, memory released\n")

