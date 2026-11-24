import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import AlterNetUNetTinyConfig, DiffusionForImageChange
import train
from utils import DiffusionSchedule


def make_model():
    config = train.tiny_config()
    model = DiffusionForImageChange(size=config.image_size, config=config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=0.03)
    return model, optimizer


def update(model, optimizer):
    x = torch.randn(2, 1, *model.size)
    schedule = DiffusionSchedule(T=4)
    return model.train_step(x, schedule.sample_timestep(2), schedule, optimizer)


def test_fresh_start_is_default():
    assert train.parse_args([]).resume is None


def test_smoke_defaults_cannot_overwrite_real_training_outputs():
    normal = train.parse_args([])
    smoke = train.parse_args(["--smoke-test"])
    assert normal.checkpoint != smoke.checkpoint
    assert normal.sample_output != smoke.sample_output
    assert "smoke" in smoke.checkpoint.parts


def test_implicit_resume_output_is_separate_from_source(tmp_path):
    source = tmp_path / "source.pt"
    args = train.parse_args(["--resume", str(source)])
    assert args.checkpoint.resolve() != source.resolve()
    assert train.parse_args(["--resume", str(source), "--checkpoint", str(source)]).checkpoint == source


def test_resume_without_output_override_preserves_source_bytes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model, optimizer = make_model()
    update(model, optimizer)
    source = tmp_path / "valuable.pt"
    model.save_model(optimizer, source, epoch=1, diffusion_config={"T": 4, "beta_start": 1e-4, "beta_end": 0.02})
    before = source.read_bytes()
    result = train.main(["--smoke-test", "--device", "cpu", "--max-batches", "1", "--resume", str(source)])
    assert source.read_bytes() == before
    assert result["completed_epochs"] == 2
    assert (tmp_path / result["checkpoint"]).is_file()


def test_real_training_updates_parameters():
    model, optimizer = make_model()
    before = model.out[-1].weight.detach().clone()
    assert math.isfinite(update(model, optimizer))
    assert not torch.equal(before, model.out[-1].weight)
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_original_default_architecture_trains_one_batch():
    model = DiffusionForImageChange()
    optimizer = torch.optim.AdamW(model.parameters())
    assert math.isfinite(update(model, optimizer))
    assert model(torch.zeros(1, 1, 32, 32), torch.tensor([1])).shape == (1, 1, 32, 32)


def test_safe_checkpoint_roundtrip_and_optimizer_resume(tmp_path):
    model, optimizer = make_model()
    update(model, optimizer)
    path = tmp_path / "nested" / "model.pt"
    schedule_config = {"T": 4, "beta_start": 1e-4, "beta_end": 0.02}
    model.save_model(optimizer, path, epoch=3, diffusion_config=schedule_config)
    raw = torch.load(path, map_location="cpu", weights_only=True)
    assert isinstance(raw["config"], dict)
    assert raw["format_version"] == 1
    loaded, resumed_optimizer, epoch = DiffusionForImageChange.load_model(path)
    assert epoch == 3
    assert loaded.diffusion_config == schedule_config
    assert resumed_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    assert resumed_optimizer.param_groups[0]["weight_decay"] == optimizer.param_groups[0]["weight_decay"]
    assert {id(p) for p in resumed_optimizer.param_groups[0]["params"]} == {id(p) for p in loaded.parameters()}
    for original, restored in zip(model.parameters(), loaded.parameters()):
        torch.testing.assert_close(original, restored, rtol=0, atol=0)
        for key in ["step", "exp_avg", "exp_avg_sq"]:
            torch.testing.assert_close(optimizer.state[original][key], resumed_optimizer.state[restored][key], rtol=0, atol=0)
    # With identical new random draws, resumed optimizer performs the same next update.
    torch.manual_seed(77)
    loss_before = update(model, optimizer)
    torch.manual_seed(77)
    loss_after = update(loaded, resumed_optimizer)
    assert loss_before == loss_after
    for original, restored in zip(model.parameters(), loaded.parameters()):
        torch.testing.assert_close(original, restored, rtol=0, atol=0)
    assert not list(path.parent.glob("*.tmp"))


def test_weights_only_inference_load(tmp_path):
    model, optimizer = make_model()
    path = tmp_path / "weights.pt"
    model.save_model(optimizer, path)
    loaded, state, epoch = DiffusionForImageChange.load_model(path, restore_optimizer=False)
    assert state is None and epoch is None
    assert not loaded.training


def test_rejects_legacy_pickle_without_unrestricted_loading(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"config": AlterNetUNetTinyConfig()}, path)
    with pytest.raises(ValueError, match="Unsafe or legacy"):
        DiffusionForImageChange.load_model(path)


def test_missing_resume_fails_before_dataset_creation(tmp_path, monkeypatch):
    def no_dataset(*args):
        raise AssertionError("Dataset must not be constructed for an invalid resume")
    monkeypatch.setattr(train, "make_loaders", no_dataset)
    with pytest.raises(FileNotFoundError, match="Checkpoint does not exist"):
        train.main(["--resume", str(tmp_path / "missing.pt")])


def test_sampler_is_finite_deterministic_and_restores_mode():
    model, _ = make_model()
    model.train()
    schedule = DiffusionSchedule(T=4)
    a = model.sample(schedule, 1, generator=torch.Generator().manual_seed(9))
    b = model.sample(schedule, 1, generator=torch.Generator().manual_seed(9))
    assert a.shape == (1, 1, 16, 16)
    assert torch.isfinite(a).all()
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert model.training


def test_evaluation_does_not_update_weights_or_mode():
    model, _ = make_model()
    loader = DataLoader(TensorDataset(torch.rand(4, 1, 16, 16), torch.zeros(4)), batch_size=2)
    before = [p.detach().clone() for p in model.parameters()]
    model.train()
    loss = train.evaluate(model, loader, DiffusionSchedule(T=4))
    assert math.isfinite(loss) and model.training
    for original, after in zip(before, model.parameters()):
        torch.testing.assert_close(original, after, rtol=0, atol=0)


def test_smoke_trains_saves_and_resumes_without_download(tmp_path, monkeypatch):
    from torchvision import datasets
    def no_download(*args, **kwargs):
        raise AssertionError("Smoke test must not construct Fashion-MNIST")
    monkeypatch.setattr(datasets, "FashionMNIST", no_download)
    checkpoint = tmp_path / "smoke.pt"
    samples = tmp_path / "samples.png"
    args = ["--smoke-test", "--device", "cpu", "--max-batches", "1", "--checkpoint", str(checkpoint), "--sample-output", str(samples)]
    first = train.main(args)
    state_before = torch.load(checkpoint, weights_only=True)["optimizer_state"]["state"]
    second = train.main(args + ["--resume", str(checkpoint)])
    state_after = torch.load(checkpoint, weights_only=True)["optimizer_state"]["state"]
    assert first["completed_epochs"] == 1 and second["completed_epochs"] == 2
    assert checkpoint.is_file() and samples.stat().st_size > 0
    for key in state_before:
        assert int(state_after[key]["step"]) == int(state_before[key]["step"]) + 1


def test_resume_rejects_conflicting_schedule_before_data(tmp_path, monkeypatch):
    model, optimizer = make_model()
    path = tmp_path / "saved.pt"
    model.save_model(optimizer, path, epoch=1, diffusion_config={"T": 4, "beta_start": 1e-4, "beta_end": 0.02})
    def no_dataset(*args):
        raise AssertionError("Dataset must not be constructed for incompatible resume")
    monkeypatch.setattr(train, "make_loaders", no_dataset)
    with pytest.raises(ValueError, match="--timesteps conflicts"):
        train.main(["--resume", str(path), "--device", "cpu", "--timesteps", "99"])


def test_validation_loader_has_no_random_augmentation(monkeypatch):
    from torchvision import datasets, transforms
    class Dataset:
        def __init__(self, root, train, download, transform):
            self.transform = transform
        def __len__(self):
            return 8
        def __getitem__(self, i):
            return torch.zeros(1, 32, 32), 0
    monkeypatch.setattr(datasets, "FashionMNIST", Dataset)
    train_loader, test_loader = train.make_loaders(train.parse_args([]), 32)
    assert any(isinstance(t, transforms.RandomRotation) for t in train_loader.dataset.transform.transforms)
    assert not any(isinstance(t, (transforms.RandomRotation, transforms.RandomHorizontalFlip)) for t in test_loader.dataset.transform.transforms)
    assert isinstance(test_loader.sampler, torch.utils.data.SequentialSampler)


def test_device_selection_and_schedule_validation(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert train.resolve_device("auto") == torch.device("cpu")
    with pytest.raises(ValueError, match="CUDA"):
        train.resolve_device("cuda")
    with pytest.raises(ValueError, match="MPS"):
        train.resolve_device("mps")
    with pytest.raises(ValueError, match="T must be positive"):
        DiffusionSchedule(T=0)
    with pytest.raises(ValueError, match="beta_start"):
        DiffusionSchedule(beta_start=-1)


def test_unsupported_optimizer_and_format_are_explicit(tmp_path):
    model, _ = make_model()
    with pytest.raises(ValueError, match="AdamW"):
        model.save_model(torch.optim.SGD(model.parameters(), lr=0.1), tmp_path / "sgd.pt")
    path = tmp_path / "unsupported.pt"
    torch.save({"format_version": 99, "config": {}}, path)
    with pytest.raises(ValueError, match="Unsupported checkpoint format"):
        DiffusionForImageChange.load_model(path)
