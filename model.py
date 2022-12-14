import pytorch_lightning as pl
import torch.nn as nn
from transformers import RobertaModel

BERT_MODEL_NAME = "roberta-base"


class QTagClassifier(pl.LightningModule):
    # Set up the classifier
    def __init__(self, n_classes=10, steps_per_epoch=None, n_epochs=3, lr=2e-5):
        super().__init__()

        self.bert = RobertaModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)
        self.type_classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size // 2),
            nn.Dropout(p=0.3),
            nn.Linear(self.bert.config.hidden_size // 2, out_features=4),
        )
        self.polarity_classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size // 2),
            nn.Dropout(p=0.3),
            nn.Linear(self.bert.config.hidden_size // 2, out_features=3),
        )
        self.tense_classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size // 2),
            nn.Dropout(p=0.3),
            nn.Linear(self.bert.config.hidden_size // 2, out_features=3),
        )
        self.certainty_classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size // 2),
            nn.Dropout(p=0.3),
            nn.Linear(self.bert.config.hidden_size // 2, out_features=2),
        )
        self.steps_per_epoch = steps_per_epoch
        self.n_epochs = n_epochs
        self.lr = lr
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attn_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attn_mask)

        return (self.type_classifier(output),
                self.polarity_classifier(output),
                self.tense_classifier(output),
                self.certainty_classifier(output))

    def training_step(self, batch, batch_size):
        output = self.bert(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

        type_pred = self.type_classifier(output)
        polarity_pred = self.polarity_classifier(output)
        tense_pred = self.tense_classifier(output)
        certainty_pred = self.certainty_classifier(output)

        pass
