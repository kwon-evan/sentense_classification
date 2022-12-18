import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import RobertaModel

BERT_MODEL_NAME = "roberta-base"


class Classifier(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.Dropout(p=0.3),
            nn.Linear(in_features // 2, out_features=4),
        )

    def forward(self, x):
        return self.classifier(x)


class RoBERTa(pl.LightningModule):
    # Set up the classifier
    def __init__(self, n_classes=10, steps_per_epoch=None, n_epochs=3, lr=2e-5):
        super().__init__()

        # LAYERS
        self.bert = RobertaModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)
        self.classifiers = {
            "type": Classifier(self.bert.config.hidden_size, 4, 0.3),
            "polarity": Classifier(self.bert.config.hidden_size, 3, 0.3),
            "tense": Classifier(self.bert.config.hidden_size, 3, 0.3),
            "certainty": Classifier(self.bert.config.hidden_size, 2, 0.3)
        }

        # PARAMS
        self.steps_per_epoch = steps_per_epoch
        self.n_epochs = n_epochs
        self.lr = lr
        self.criterion = {
            "type": nn.CrossEntropyLoss(),
            "polarity": nn.CrossEntropyLoss(),
            "tense": nn.CrossEntropyLoss(),
            "certainty": nn.CrossEntropyLoss()
        }

    def forward(self, input_ids, attn_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attn_mask)

        return {
            "type": self.classifiers["type"](output),
            "polarity": self.classifiers["polarity"](output),
            "tense": self.classifiers["tense"](output),
            "certainty": self.classifiers["certainty"](output)
        }

    def training_step(self, batch, batch_size):
        outputs = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

        losses = {k: self.criterion[k](batch[k], v) for k, v in outputs.items()}

        self.log_dict(losses, prog_bar=True, logger=True)

        return sum(losses.values()) / len(losses)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
