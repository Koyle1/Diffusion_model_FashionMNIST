import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils import DiffusionSchedule
from model import DiffusionForImageChange

torch.manual_seed(42)

def main():
    ## config
    BATCH_SIZE = 128
    IMAGE_SIZE = 32
    NUM_WORKERS = 0
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    DIAGNOSTIC = False
    TEST = False
    Load = True

    ## Transformation (added simple augmentations)
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    ## Dataset
    train_dataset = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    # Data_loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    schedule = DiffusionSchedule(T=200, device=DEVICE)
    if Load:
        model, optimizer, epoch = DiffusionForImageChange.load_model("diffusion_model.pth", device=DEVICE)
    else:
        model = DiffusionForImageChange(size=(IMAGE_SIZE, IMAGE_SIZE)).to(DEVICE)
        model.reset_parameters()
        model = model.float()
        model.debug_diagnostics = DIAGNOSTIC

    # add weight decay for regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2, eps=1e-6)

    if DIAGNOSTIC:
        # Run a single debug batch to trigger diagnostics quickly
        it = iter(train_loader)
        x0, _ = next(it)
        x0 = x0.to(DEVICE)
        t = schedule.sample_timestep(x0.shape[0]).to(DEVICE) 
        try:
            loss = model.train_step(x0, t, schedule, optimizer)
            print("DEBUG_STEP: loss=", loss)
        except Exception as e:
            print("DEBUG_STEP ABORTED:", e)
        return
    if TEST:
        # Run single batch for testing
        x0, _ = next(iter(train_loader))
        x0 = x0.to(DEVICE)

        for i in range(400):
            t = schedule.sample_timestep(x0.size(0)).to(DEVICE)
            loss = model.train_step(x0, t, schedule, optimizer)
            if i % 50 == 0:
                print(i, loss)
    else:
        model.train_model(train_loader, schedule, optimizer, epochs=20)
        model.save_model(model, optimizer, path="diffusion_model.pth", epoch=20)

if __name__ == "__main__":
    main()

    


