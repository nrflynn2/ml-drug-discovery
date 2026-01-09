import datetime
import torch
import typer
import json
import functools
import pandas as pd
import numpy as np

from pathlib import Path
from typing_extensions import Annotated
from src.model import AntibodyClassifier
from src.data import Tokenizer, load_data, BCRDataset, collate_fn
from src.utils import set_seeds, get_device
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_recall_fscore_support,
)

# create a typer app
app = typer.Typer()


class AntibodyPredictor:
    """A class for making predictions using a trained AntibodyClassifier model."""

    def __init__(self, model: AntibodyClassifier, device: torch.device):
        """Initializes the AntibodyPredictor.

        Args:
            model (AntibodyClassifier): A trained AntibodyClassifier model.
            device (torch.device): The device (CPU or GPU) to use for inference.
        """

        self.model = model
        self.model.eval()
        self.model.to(device)

    def __call__(
        self, test_dl: torch.utils.data.DataLoader
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Makes predictions on a test dataset.

        Args:
            test_dl (torch.utils.data.DataLoader): A DataLoader containing the test data.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - y_true: Ground truth labels
                - y_pred: Predicted labels
                - y_prob: Class probabilities
        """

        y_true, y_pred, y_prob = [], [], []
        with torch.inference_mode():
            for batch in test_dl:
                logits = self.model(batch)

                # truth, and predictions for each batch
                y_true.append(batch["label"].cpu().numpy())
                y_pred.append(torch.argmax(logits, dim=-1).cpu().numpy())
                y_prob.append(torch.softmax(logits, dim=-1).cpu().numpy())

        return (
            np.concatenate(y_true, axis=None),
            np.concatenate(y_pred, axis=None),
            np.vstack(y_prob),
        )

    @classmethod
    def from_run_dir(cls, run_dir: Path, device: torch.device) -> "AntibodyPredictor":
        """Loads a trained AntibodyPredictor from saved files associated with a run ID

        Args:
            run_dir (Path): The path to the directory containing the saved model and training artifacts.
            device (torch.device): The device (CPU or GPU) to use for inference.

        Returns:
            AntibodyPredictor: An initialized AntibodyPredictor instance.
        """

        with open(run_dir / "args.json", "r") as f:
            kwargs = json.load(f)
        model = AntibodyClassifier(**kwargs)
        best_model_path = run_dir / "best_model.pt"
        model.load_state_dict(torch.load(best_model_path))
        return cls(model=model, device=device)


@app.command()
def evaluate(
    run_dir: Annotated[
        str,
        typer.Option(help="Path to the output directory for a training or tuning run"),
    ],
    dataset_loc: Annotated[
        str, typer.Option(help="Path to the test dataset in parquet format")
    ],
    batch_size: Annotated[int, typer.Option(help="Number of samples per batch")] = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Evaluates a trained model on a test dataset.

    Args:
        run_dir (str): Path to the output directory for a training or tuning run.
        dataset_loc (str): Path to the test dataset in parquet format.
        batch_size (int, optional): Number of samples per batch. Defaults to 64.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
            - y_true: Ground-truth labels (NumPy array)
            - y_pred: Predicted labels (NumPy array)
            - y_prob: Class probabilities (NumPy array)
            - metrics: A dictionary containing evaluation metrics.
    """
    run_dir = Path(run_dir)

    # load test data
    df, _ = load_data(dataset_loc)
    df.reset_index(inplace=True, drop=True)
    test_ds = BCRDataset(df)

    tokenizer = Tokenizer()
    device = get_device()

    # test dataloader
    collate_fn_partial = functools.partial(
        collate_fn, tokenizer=tokenizer, device=device
    )
    test_dl = DataLoader(test_ds, collate_fn=collate_fn_partial, batch_size=batch_size)

    # load model
    predictor = AntibodyPredictor.from_run_dir(run_dir, device)

    y_true, y_pred, y_prob = predictor(test_dl)

    # Evaluation metrics
    metrics = {}

    # accuracy
    metrics["accuracy"] = accuracy_score(y_true, y_pred)

    # auc score
    if y_prob.shape[1] == 2:
        auc_score = roc_auc_score(y_true, y_prob[:, 1])
        metrics["auc_score"] = auc_score

    metrics["precision"], metrics["recall"], metrics["f1"], _ = (
        precision_recall_fscore_support(y_true, y_pred, average="weighted")
    )

    # Save evaluation metrics
    with open(run_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4, sort_keys=False)

    return y_true, y_pred, y_prob, metrics


if __name__ == "__main__":
    set_seeds()
    app()
