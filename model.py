import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import AutoModel


class Classifier(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.Dropout(p=dropout),
            nn.Linear(in_features // 2, out_features),
        )

    def forward(self, x):
        return self.classifier(x)


class RoBERTa(pl.LightningModule):
    # Set up the classifier
    def __init__(self, model_name: str = "roberta-base", lr=2e-5):
        super().__init__()

        # LAYERS
        self.roberta = AutoModel.from_pretrained(model_name, return_dict=True)
        self.type_classifier = nn.Linear(self.roberta.config.hidden_size, 4)
        self.polarity_classifier = nn.Linear(self.roberta.config.hidden_size, 3)
        self.tense_classifier = nn.Linear(self.roberta.config.hidden_size, 3)
        self.certainty_classifier = nn.Linear(self.roberta.config.hidden_size, 2)

        # PARAMS
        self.lr = lr
        self.criterion = {
            "type": nn.CrossEntropyLoss(),
            "polarity": nn.CrossEntropyLoss(),
            "tense": nn.CrossEntropyLoss(),
            "certainty": nn.CrossEntropyLoss(),
        }

    def forward(self, input_ids, attn_mask):
        output = self.roberta(input_ids=input_ids, attention_mask=attn_mask)

        return {
            "type": self.type_classifier(output.pooler_output),
            "polarity": self.polarity_classifier(output.pooler_output),
            "tense": self.tense_classifier(output.pooler_output),
            "certainty": self.certainty_classifier(output.pooler_output)
        }

    def training_step(self, batch, batch_size):
        outputs = self(batch["input_ids"], batch["attention_mask"])

        losses = {f'train/loss-{k}': self.criterion[k](v, batch[k]) for k, v in outputs.items()}
        total_loss = sum(losses.values()) / len(losses)
        losses['train/loss-total'] = total_loss

        self.log_dict(losses, prog_bar=True, logger=True)
        return total_loss

    def validation_step(self, batch, batch_size):
        outputs = self(batch["input_ids"], batch["attention_mask"])

        losses = {f'val/loss-{k}': self.criterion[k](v, batch[k]) for k, v in outputs.items()}
        total_loss = sum(losses.values()) / len(losses)
        losses['val/loss-total'] = total_loss

        self.log_dict(losses, prog_bar=True, logger=True)

    def predict_step(self, batch, batch_size):
        outputs = self(batch["input_ids"], batch["attention_mask"])
        return outputs

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
