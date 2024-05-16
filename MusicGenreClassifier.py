import lightning
import torch
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F


class MusicClassifier(lightning.LightningModule):

    def __init__(self, model):
        super().__init__()
        self.model: nn.Module = model
        self.lr: float = 0.0002
        self.num_classes: int = 10
        self.current_epoch_training_loss = torch.tensor(0.0)
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes)

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, x, y):
        return F.cross_entropy(x, y)

    def common_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.compute_loss(outputs, y)
        return loss, outputs, y

    def training_step(self, batch, batch_idx):
        loss, outputs, y = self.common_step(batch, batch_idx)
        self.training_step_outputs.append(loss)
        acc = self.accuracy(outputs, y)
        self.log_dict(
            {"train_loss": loss, "train_accuracy": acc},
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )
        return {'loss': loss}

    def on_train_epoch_end(self):
        outs = torch.stack(self.training_step_outputs)
        self.current_epoch_training_loss = outs.mean()
        self.training_step_outputs.clear()

    def common_validation_and_test_step(self, batch, batch_idx):
        loss, outputs, y = self.common_step(batch, batch_idx)
        acc = self.accuracy(outputs, y)
        return loss, acc

    def validation_step(self, batch, batch_idx):
        loss, acc = self.common_validation_and_test_step(batch, batch_idx)
        self.validation_step_outputs.append(loss)
        self.log_dict(
            {"val_loss": loss, "val_acc": acc},
            on_step=True,
            on_epoch=True,
            prog_bar=True
        )
        return {'val_loss': loss, 'val_acc': acc}

    def on_validation_epoch_end(self):
        outs = torch.stack(self.validation_step_outputs)
        avg_loss = outs.mean()
        self.logger.experiment.add_scalars('train and vall losses',
                                           {'train': self.current_epoch_training_loss.item(), 'val': avg_loss.item()},
                                           self.current_epoch)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        loss, acc = self.common_validation_and_test_step(batch, batch_idx)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'test_loss': loss, 'test_acc': acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [lr_scheduler]
