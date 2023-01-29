import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)

from config import CHECKPOINTS_DIR, LOGS_DIR


class LightningModel(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 1e-3,
        loss_cls=nn.BCEWithLogitsLoss,
    ) -> None:
        super().__init__()

        self.loss_fn = loss_cls()
        self.lr = learning_rate

        self.f1_train = BinaryF1Score()
        self.f1_test = BinaryF1Score()

        self.acc_train = BinaryAccuracy()
        self.acc_test = BinaryAccuracy()

        self.precision_train = BinaryPrecision()
        self.precision_test = BinaryPrecision()

        self.recall_train = BinaryRecall()
        self.recall_test = BinaryRecall()

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(),
            lr=self.lr,
            # weight_decay=5e-4,
        )

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, y = batch
        preds = self(x)

        loss = self.loss_fn(preds, y.float())
        self.f1_train(preds, y)
        self.acc_train(preds, y)
        self.precision_train(preds, y)
        self.recall_train(preds, y)

        self.log("train/loss", loss, on_epoch=True, on_step=False)
        self.log("train/f1", self.f1_train, on_epoch=True, on_step=False)
        self.log("train/acc", self.acc_train, on_epoch=True, on_step=False)
        self.log("train/precision", self.precision_train, on_epoch=True, on_step=False)
        self.log("train/recall", self.recall_train, on_epoch=True, on_step=False)

        return loss

    # This must be a validation_step or the ModelCheckpoint won't pick up the metrics
    def validation_step(self, batch, batch_idx: int):
        x, y = batch
        preds = self(x)

        loss = self.loss_fn(preds, y.float())
        self.f1_test(preds, y)
        self.acc_test(preds, y)
        self.precision_test(preds, y)
        self.recall_test(preds, y)

        self.log("test/loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("test/f1", self.f1_test, on_epoch=True, on_step=False, prog_bar=True)
        self.log("test/acc", self.acc_test, on_epoch=True, on_step=False, prog_bar=True)
        self.log("test/precision", self.precision_test, on_epoch=True, on_step=False)
        self.log("test/recall", self.recall_test, on_epoch=True, on_step=False)


class BinaryMLP(LightningModel):
    def __init__(
        self,
        emb_dim: int,
        hidden_dims: list[int],
        learning_rate: float = 1e-3,
        loss_cls=nn.BCEWithLogitsLoss,
    ) -> None:
        super().__init__(learning_rate=learning_rate, loss_cls=loss_cls)

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout(0.3),
            *(
                layer
                for i in range(len(hidden_dims) - 1)
                for layer in (
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                )
            ),
            nn.Linear(hidden_dims[-1], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze()


class LSTMModel(LightningModel):
    def __init__(self, feature_size, hidden_size, num_layers=3, batch_first=True, learning_rate: float = 1e-3, loss_cls=nn.BCELoss):
        super().__init__(learning_rate, loss_cls)
        self.model = torch.nn.ModuleDict({
            'lstm': torch.nn.LSTM(
                input_size=feature_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first
            ),
            'linear': nn.Sequential(
                nn.Linear(in_features=hidden_size, out_features=1),
                nn.Sigmoid()
            )
            
        })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.model['lstm'](x)
        linear_in = hidden[0][-1]
        return self.model['linear'](linear_in).squeeze()


class RNNModel(LightningModel):
    def __init__(self, feature_size, hidden_size, num_layers=3, batch_first=True, learning_rate: float = 1e-3, loss_cls=nn.BCELoss):
        super().__init__(learning_rate, loss_cls)
        self.model = torch.nn.ModuleDict({
            'lstm': nn.RNN(
                input_size=feature_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first
            ),
            'linear': nn.Sequential(
                nn.Linear(in_features=hidden_size, out_features=1),
                nn.Sigmoid()
            )
            
        })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.model['lstm'](x)
        linear_in = hidden[-1]
        return self.model['linear'](linear_in).squeeze()

class CNNModel(LightningModel):
    def __init__(
        self,
        emb_dim: int = 300,
        conv_kernels: list[int] = [3, 4, 5],
        conv_filter: int = 100,
        head_dim: int = 128,
        sentence_length: int = 32,
        learning_rate: float = 1e-3,
        loss_cls=nn.BCEWithLogitsLoss,
    ) -> None:
        super().__init__(learning_rate, loss_cls)

        self.convs = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(1, conv_filter, (kernel, emb_dim)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((sentence_length - kernel + 1, 1)),
            )
            for kernel in conv_kernels
        )

        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(conv_filter * len(conv_kernels), head_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(head_dim),
            nn.Linear(head_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)

        out_convs = [conv(x) for conv in self.convs]
        out = torch.cat(out_convs, dim=1).squeeze()
        out = self.head(out).squeeze()

        return out


class MetricTracker(Callback):
    def __init__(self):
        self.best_epoch = {}
        self.best_loss = float("inf")

    def on_validation_epoch_end(
        self,
        trainer: "pl.Trainer",
        module: "pl.LightningModule",
    ) -> None:
        assert isinstance(trainer.logged_metrics["test/loss"], torch.Tensor)
        loss = trainer.logged_metrics["test/loss"].cpu().item()

        if loss < self.best_loss:
            self.best_loss = loss
            self.best_epoch = trainer.logged_metrics.copy()


def train_model(
    model: LightningModel,
    datamodule: pl.LightningDataModule,
    epochs: int = 50,
    name: str = "model",
    gpu: bool = True,
    verbose: bool = True,
) -> dict[str, float]:
    metric_tracker = MetricTracker()

    model_chkpt = ModelCheckpoint(
        dirpath=CHECKPOINTS_DIR,
        filename=name,
        monitor="test/loss",
        mode="min",
        verbose=verbose,
    )

    early_stopping = EarlyStopping(
        monitor="test/loss",
        mode="min",
        patience=10,
        strict=True,
        check_finite=True,
    )

    trainer = pl.Trainer(
        logger=TensorBoardLogger(
            save_dir=LOGS_DIR,
            name=name,
            default_hp_metric=False,
        ),
        callbacks=[model_chkpt, early_stopping, metric_tracker],
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        max_epochs=epochs,
        accelerator="gpu" if gpu else "cpu",
        enable_progress_bar=verbose,
        enable_model_summary=verbose,
    )

    trainer.fit(model=model, datamodule=datamodule)

    return {k: v.cpu().item() for k, v in metric_tracker.best_epoch.items()}
