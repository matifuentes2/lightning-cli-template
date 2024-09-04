import torch.nn as nn
import lightning as L
import torchmetrics
from src.models.example_model import ExampleModel
import torch.nn as nn
import lightning as L
import torchmetrics
from src.models.example_model import ExampleModel

# Define the MLP model using PyTorch Lightning


class ExampleModule(L.LightningModule):
    def __init__(self, model_params, criterion):
        super(ExampleModule, self).__init__()
        self.model = ExampleModel(**model_params)
        self.criterion = criterion

        # Metrics for binary classification
        self.accuracy = torchmetrics.Accuracy(task="binary")
        self.precision = torchmetrics.Precision(task="binary")
        self.recall = torchmetrics.Recall(task="binary")
        self.f1_score = torchmetrics.F1Score(task="binary")
        self.auc = torchmetrics.AUROC(task="binary")

        self.metrics = nn.ModuleDict({
            'accuracy': torchmetrics.Accuracy(task="binary"),
            'precision': torchmetrics.Precision(task="binary"),
            'recall': torchmetrics.Recall(task="binary"),
            'f1_score': torchmetrics.F1Score(task="binary"),
            'auc': torchmetrics.AUROC(task="binary")
        })

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, batch_idx, stage):
        adj, labels = batch
        adj = adj.view(adj.size(0), -1)  # Flatten the adjacency matrices
        outputs = self(adj).squeeze()
        loss = self.criterion(outputs, labels.float())

        self.log(f'{stage}_loss', loss, on_step=(
            stage == 'train'), on_epoch=True, prog_bar=True)

        for metric_name, metric in self.metrics.items():
            value = metric(outputs, labels.int())
            self.log(f'{stage}_{metric_name}', value, on_step=(
                stage == 'train'), on_epoch=True, prog_bar=(metric_name == 'accuracy'))

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, batch_idx, "test")
