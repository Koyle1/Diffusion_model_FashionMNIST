"""Train from scratch by default; resume only when explicitly requested."""

import argparse
import json
from pathlib import Path
import uuid

import torch
from torch.utils.data import DataLoader, TensorDataset

from model import AlterNetUNetTinyConfig, DiffusionForImageChange
from utils import DiffusionSchedule


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, help="explicitly restore a version-1 checkpoint")
    parser.add_argument("--checkpoint", type=Path, help="output checkpoint; smoke defaults are separate from real training")
    parser.add_argument("--sample-output", type=Path, help="output PNG; defaults beside the checkpoint")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--epochs", type=positive_int, help="additional epochs; defaults to 20, or 1 in smoke mode")
    parser.add_argument("--batch-size", type=positive_int, default=128)
    parser.add_argument("--image-size", type=positive_int, help="defaults to 32, or 16 in smoke mode")
    parser.add_argument("--timesteps", type=positive_int, help="defaults to 200, or 4 in smoke mode")
    parser.add_argument("--max-batches", type=positive_int, help="optional cap on batches per train/evaluation epoch")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threads", type=positive_int)
    parser.add_argument("--lr", type=float, default=3e-4, help="fresh-start learning rate; resume restores saved optimizer values")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="fresh-start weight decay; resume restores saved optimizer values")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=positive_int, default=16)
    parser.add_argument("--no-download", action="store_true", help="require an existing Fashion-MNIST dataset")
    parser.add_argument("--smoke-test", action="store_true", help="tiny synthetic-data integration check; no download or quality benchmark")
    args = parser.parse_args(argv)
    if args.num_workers < 0:
        parser.error("--num-workers must not be negative")
    if args.lr <= 0 or args.weight_decay < 0:
        parser.error("--lr must be positive and --weight-decay nonnegative")
    if args.checkpoint is None:
        if args.resume is not None:
            # An implicit output must never overwrite the resume source.
            group = "smoke" if args.smoke_test else "resume"
            args.checkpoint = Path("artifacts") / group / f"{args.resume.stem}-{uuid.uuid4().hex}.pt"
        else:
            directory = Path("artifacts/smoke") if args.smoke_test else Path("artifacts")
            args.checkpoint = directory / "diffusion_model.pt"
    if args.sample_output is None:
        args.sample_output = args.checkpoint.with_name(args.checkpoint.stem + "_samples.png")
    return args


def resolve_device(requested):
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return torch.device(requested)


def tiny_config(image_size=16):
    return AlterNetUNetTinyConfig(image_size=(image_size, image_size), time_dim=8,
                                 c1=8, c2=8, c3=8, c4=8, h2=1, h3=1, h4=1)


def make_loaders(args, image_size):
    generator = torch.Generator().manual_seed(args.seed)
    if args.smoke_test:
        train_x = torch.rand((8, 1, image_size, image_size), generator=generator) * 2 - 1
        test_x = torch.rand((4, 1, image_size, image_size), generator=generator) * 2 - 1
        train_data = TensorDataset(train_x, torch.zeros(8, dtype=torch.long))
        test_data = TensorDataset(test_x, torch.zeros(4, dtype=torch.long))
        batch_size = 2
    else:
        # Import lazily; smoke mode never constructs or downloads Fashion-MNIST.
        from torchvision import datasets, transforms
        common = [transforms.Resize(image_size), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        train_transform = transforms.Compose([transforms.Resize(image_size), transforms.RandomHorizontalFlip(),
                                              transforms.RandomRotation(10), *common[1:]])
        test_transform = transforms.Compose(common)
        train_data = datasets.FashionMNIST(root=args.data_dir, train=True, download=not args.no_download,
                                          transform=train_transform)
        test_data = datasets.FashionMNIST(root=args.data_dir, train=False, download=not args.no_download,
                                         transform=test_transform)
        batch_size = args.batch_size
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=args.num_workers, generator=generator)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    return train_loader, test_loader


def train_epoch(model, loader, schedule, optimizer, max_batches=None):
    model.train()
    device = next(model.parameters()).device
    total, examples = 0.0, 0
    for index, (x, _) in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        x = x.to(device=device, dtype=torch.float32)
        loss = model.train_step(x, schedule.sample_timestep(len(x)), schedule, optimizer)
        if not torch.isfinite(torch.tensor(loss)):
            raise FloatingPointError("Nonfinite training loss")
        total += loss * len(x)
        examples += len(x)
    if not examples:
        raise ValueError("Training loader is empty")
    return total / examples


@torch.no_grad()
def evaluate(model, loader, schedule, max_batches=None):
    """Held-out noise-prediction MSE, not an image-quality or discovery score."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    total, examples = 0.0, 0
    try:
        for index, (x, _) in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            x = x.to(device=device, dtype=torch.float32)
            loss = model.compute_loss(x, schedule.sample_timestep(len(x)), schedule)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite evaluation loss")
            total += float(loss) * len(x)
            examples += len(x)
        if not examples:
            raise ValueError("Evaluation loader is empty")
        return total / examples
    finally:
        model.train(was_training)


def main(argv=None):
    args = parse_args(argv)
    # Reject missing resume before constructing/downloading any dataset.
    if args.resume is not None and not args.resume.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.resume}")
    if args.threads is not None or args.smoke_test:
        torch.set_num_threads(args.threads or 1)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    if args.resume is not None:
        model, optimizer, completed = DiffusionForImageChange.load_model(args.resume, str(device))
        if optimizer is None:
            raise ValueError("Resume requires a checkpoint containing AdamW state")
        if not model.diffusion_config:
            raise ValueError("Resume requires saved diffusion schedule configuration")
        if args.image_size is not None and (args.image_size, args.image_size) != tuple(model.size):
            raise ValueError("--image-size conflicts with the checkpoint architecture")
        if args.timesteps is not None and args.timesteps != model.diffusion_config["T"]:
            raise ValueError("--timesteps conflicts with the checkpoint schedule")
        schedule_config = model.diffusion_config
        completed = completed or 0
    else:
        image_size = args.image_size or (16 if args.smoke_test else 32)
        if image_size < 16:
            raise ValueError("--image-size must be at least 16")
        config = tiny_config(image_size) if args.smoke_test else AlterNetUNetTinyConfig(image_size=(image_size, image_size))
        model = DiffusionForImageChange(size=config.image_size, config=config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-6)
        schedule_config = {"T": args.timesteps or (4 if args.smoke_test else 200), "beta_start": 1e-4, "beta_end": 0.02}
        completed = 0
    schedule = DiffusionSchedule(device=device, **schedule_config)
    train_loader, test_loader = make_loaders(args, model.size[0])
    epochs = args.epochs or (1 if args.smoke_test else 20)
    max_batches = args.max_batches or (2 if args.smoke_test else None)
    for epoch in range(completed, completed + epochs):
        training_loss = train_epoch(model, train_loader, schedule, optimizer, max_batches)
        evaluation_loss = evaluate(model, test_loader, schedule, max_batches)
        model.save_model(optimizer, args.checkpoint, epoch=epoch + 1, diffusion_config=schedule_config)
        print(json.dumps({"epoch": epoch + 1, "train_noise_mse": training_loss, "heldout_noise_mse": evaluation_loss}))
    samples = model.sample(schedule, 2 if args.smoke_test else args.samples)
    if not torch.isfinite(samples).all():
        raise FloatingPointError("Sampler produced nonfinite values")
    from torchvision.utils import save_image
    args.sample_output.parent.mkdir(parents=True, exist_ok=True)
    save_image(samples.cpu(), args.sample_output, normalize=True, value_range=(-1, 1))
    summary = {"completed_epochs": completed + epochs, "checkpoint": str(args.checkpoint),
               "samples": str(args.sample_output), "device": str(device), "smoke_test": args.smoke_test,
               "train_noise_mse": training_loss, "heldout_noise_mse": evaluation_loss}
    print(json.dumps(summary, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
