from typing import Optional

import pandas as pd
from pandas import DataFrame
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split


class RoBertaDataset(Dataset):
    def __init__(self, df: DataFrame, is_train: bool, tokenizer, max_len: int):
        self.sentence = df.sentence.values
        self.labels = {
            "type": df.type.values,
            "polarity": df.polarity.values,
            "tense": df.tense.values,
            "certainty": df.certainty.values,
        } if is_train else None

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, index):
        sentence = self.sentence[index]
        inputs = self.tokenizer.encode_plus(
            sentence,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=False,
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].flatten()
        attn_mask = inputs['attention_mask'].flatten()

        if self.labels is not None:
            return {
                'input_ids': input_ids,
                'attention_mask': attn_mask,
                "type": torch.tensor(self.labels["type"], dtype=torch.float),
                "polarity": torch.tensor(self.labels["polarity"], dtype=torch.float),
                "tense": torch.tensor(self.labels["tense"], dtype=torch.float),
                "certainty": torch.tensor(self.labels["certainty"], dtype=torch.float)
            }
        else:
            return {
                'input_ids': input_ids,
                'attention_mask': attn_mask,
            }


class RoBertaDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size: int,
                 tokenizer,
                 max_token_len: int,
                 train_df: pd.DataFrame = None,
                 predict_df: pd.DataFrame = None
                 ):
        super.__init__()
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.train_df = train_df
        self.predict_df = predict_df

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit' or stage == 'train':
            train, valid = train_test_split(self.train_df, test_size=0.2)
            self.train = RoBertaDataset(train, True, self.tokenizer, self.max_token_len)
            self.val = RoBertaDataset(valid, True, self.tokenizer, self.max_token_len)

        if stage == 'predict':
            self.predict = RoBertaDataset(self.predict_df, False, self.tokenizer, self.max_token_len)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=4)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=self.batch_size, num_workers=4)
