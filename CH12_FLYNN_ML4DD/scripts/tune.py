import torch
import torch.nn as nn
import tempfile
import functools
import ray
import typer
import json

from pathlib import Path
from ray import train, tune
from typing_extensions import Annotated
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from src.utils import set_seeds, get_device
from src.model import AntibodyClassifier
from scripts.train import get_dataloaders, train_epoch, val_epoch
from src.data import Tokenizer, load_data
from typing import Any


# create a typer app
app = typer.Typer()


def train_loop(config: dict[str, Any], dataset_loc: str) -> None:
    """Trains a model using the provided configuration and dataset.

    Args:
        config (Dict[str, Any]): A dictionary containing model and training hyperparameters.
        dataset_loc (str): Path to the dataset in parquet format.
    """

    # Dataset
    df, classes = load_data(dataset_loc)
    tokenizer = Tokenizer()
    device = get_device()

    # train and val dataloaders
    train_dl, val_dl = get_dataloaders(
        df, config["val_size"], tokenizer, device, config["batch_size"]
    )

    model_kwargs = {
        "vocab_size": tokenizer.vocab_size,
        "padding_idx": tokenizer.pad_token_id,
        "embedding_dim": config["embedding_dim"],
        "num_layers": config["num_layers"],
        "num_heads": config["num_heads"],
        "ffn_dim": config["embedding_dim"] * 2,
        "dropout": config["dropout"],
        "num_classes": config["num_classes"],
    }

    # model
    model = AntibodyClassifier(**model_kwargs)

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"])

    start = 1
    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_dict = torch.load(Path(checkpoint_dir) / "checkpoint.pt")
            start = checkpoint_dict["epoch"] + 1
            model.load_state_dict(checkpoint_dict["model_state"])

    model.to(device)

    for epoch in range(start, config["num_epochs"] + 1):
        # Train
        train_loss = train_epoch(model, train_dl, loss_fn, opt)
        # Validation
        val_loss, _, _, _ = val_epoch(model, val_dl, loss_fn)

        metrics = {"train_loss": train_loss, "val_loss": val_loss}
        with tempfile.TemporaryDirectory() as tempdir:
            checkpoint_dict = {
                "classes": classes,
                "model_kwargs": model_kwargs,
                "epoch": epoch,
                "model_state": model.state_dict(),
            }
            torch.save(
                checkpoint_dict,
                Path(tempdir) / "checkpoint.pt",
            )
            train.report(metrics=metrics, checkpoint=Checkpoint.from_directory(tempdir))


@app.command()
def tune_model(
    run_id: Annotated[str, typer.Option(help="Name for the training run ID")],
    dataset_loc: Annotated[
        str, typer.Option(help="Absolute path to the dataset in parquet format")
    ],
    val_size: Annotated[
        float, typer.Option(help="Proportion of the dataset to use for validation")
    ] = 0.15,
    num_classes: Annotated[
        int, typer.Option(help="Number of final output dimensions")
    ] = 2,
    batch_size: Annotated[int, typer.Option(help="Number of samples per batch")] = 32,
    num_epochs: Annotated[int, typer.Option(help="Number of epochs for training")] = 30,
    num_samples: Annotated[int, typer.Option(help="Number of trials for tuning")] = 100,
    gpu_per_trial: Annotated[float, typer.Option(help="Number of GPU per trial")] = 0.2,
    output_dir: Annotated[
        str, typer.Option(help="Path to save the best model and tuning results")
    ] = "runs",
) -> tune.ResultGrid:
    """Tunes a model using Ray Tune and saves the best result.

    Args:
        run_id (str, optional): Name for the training run ID to track results.
        dataset_loc (str, optional): Absolute path to the dataset in parquet format.
        val_size (float, optional): Proportion of the dataset to use for validation.
                                    Defaults to 0.15.
        num_classes (int, optional): Number of final output dimensions. Defaults to 2.
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        num_epochs (int, optional): Number of epochs for training. Defaults to 30.
        num_samples (int, optional): Number of trials for tuning. Defaults to 100.
        gpu_per_trial (float, optional): Number of GPUs per trial. Defaults to 0.2.
        output_dir (str): Path to save the best model and tuning results.

    Returns:
        tune.ExperimentAnalysis: Ray Tune object containing analysis of the tuning process.
    """

    config = {
        "embedding_dim": tune.choice([2**i for i in range(4, 8)]),  # 16, 32, 64, 128
        "num_layers": tune.choice([i for i in range(1, 9)]),  # 1 to 8
        "num_heads": tune.choice([1, 2, 4, 8]),  # 1, 2, 4, 8
        "dropout": tune.quniform(0, 0.2, 0.02),
        "lr": tune.loguniform(1e-5, 1e-3),
        "num_classes": num_classes,
        "val_size": val_size,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
    }

    # early stopping with adaptive successive halving
    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
    )

    trainable = functools.partial(train_loop, dataset_loc=dataset_loc)
    # make sure GPU is available
    device = get_device()
    if device.type == "cuda":
        trainable = tune.with_resources(trainable, {"gpu": gpu_per_trial})
    tuner = tune.Tuner(
        trainable,
        tune_config=tune.TuneConfig(num_samples=num_samples, scheduler=scheduler),
        param_space=config,
    )
    results = tuner.fit()

    save_path = Path(f"{output_dir}/{run_id}")
    if not save_path.exists():
        save_path.mkdir(parents=True)

    # save tune results
    results_df = results.get_dataframe(filter_metric="val_loss", filter_mode="min")
    results_df.to_csv(save_path / "tune_results.csv", index=False)

    # save best model and params
    best_result = results.get_best_result("val_loss", mode="min")
    with best_result.checkpoint.as_directory() as checkpoint_dir:
        checkpoint_dict = torch.load(Path(checkpoint_dir) / "checkpoint.pt")

        # save model_state to save_path
        model_state = checkpoint_dict["model_state"]
        torch.save(model_state, save_path / f"best_model.pt")

        # save model parameters
        with open(save_path / "args.json", "w") as f:
            json.dump(checkpoint_dict["model_kwargs"], f, indent=4, sort_keys=False)

        # save classes
        with open(save_path / "classes.json", "w") as f:
            json.dump(checkpoint_dict["classes"], f, indent=4, sort_keys=False)
    return results


if __name__ == "__main__":
    set_seeds()
    if ray.is_initialized():
        ray.shutdown()
    ray.init()
    app()
