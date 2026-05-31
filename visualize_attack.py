import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack
from models.simple_cnn import SimpleCNN
from utils.visualize import show_adversarial_example


def load_test_batch(data_dir, batch_size, device):
    transform = transforms.ToTensor()
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    images, labels = next(iter(test_loader))
    return images.to(device), labels.to(device)


def save_attack_visualization(model, images, labels, attack_fn, output_path, **attack_kwargs):
    adv_images = attack_fn(model, images, labels, **attack_kwargs)
    show_adversarial_example(
        images[0],
        adv_images[0],
        save_path=output_path,
        show=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Save FGSM and PGD visual examples.")
    parser.add_argument("--model-path", type=str, default="simple_cnn_mnist.pth")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    images, labels = load_test_batch(args.data_dir, args.batch_size, device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fgsm_path = output_dir / "fgsm_eps_0.3_example.png"
    pgd_path = output_dir / "pgd_eps_0.3_example.png"

    save_attack_visualization(
        model,
        images,
        labels,
        fgsm_attack,
        fgsm_path,
        epsilon=0.3,
    )
    print(f"Saved {fgsm_path}")

    save_attack_visualization(
        model,
        images,
        labels,
        pgd_attack,
        pgd_path,
        epsilon=0.3,
        alpha=0.01,
        steps=40,
    )
    print(f"Saved {pgd_path}")


if __name__ == "__main__":
    main()
