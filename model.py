import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.functional.classification import multiclass_hamming_distance
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
        self.tense_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.roberta.config.hidden_size, 3)
        )
        self.type_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.roberta.config.hidden_size + 3, 4),
        )
        self.polarity_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.roberta.config.hidden_size + 3 + 4, 3)
        )
        self.certainty_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.roberta.config.hidden_size + 3 + 4 + 3, 2)
        )

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
        z = output.pooler_output

        # type: # 1, polarity: 2, tense: 3, certainty: 4
        # 3 -> 1 -> 2 -> 4
        y_1 = self.tense_classifier(z, 3)
        y_2 = self.type_classifier(torch.cat((z + y_1), dim=1), 4)
        y_3 = self.polarity_classifier(torch.cat((z + y_1 + y_2), dim=1), 3)
        y_4 = self.certainty_classifier(torch.cat((z + y_1 + y_2 + y_3), dim=1), 2)

        print(f"y_1: {y_1.shape}")
        print(f"y_2: {y_2.shape}")
        print(f"y_3: {y_3.shape}")
        print(f"y_4: {y_4.shape}")

        return {
            "type": y_2,
            "polarity": y_3,
            "tense": y_1,
            "certainty": y_4
        }

        # return {
        #     "type": self.type_classifier(output.pooler_output),
        #     "polarity": self.polarity_classifier(output.pooler_output),
        #     "tense": self.tense_classifier(output.pooler_output),
        #     "certainty": self.certainty_classifier(output.pooler_output)
        # }

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
