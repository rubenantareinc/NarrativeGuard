from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)


@dataclass
class BERTPrediction:
    label: int
    score: float


class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class BERTBaseline:
    """Neural baseline using BERT-base with minimal fine-tuning."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = 256,
        lr: float = 2e-5,
        batch_size: int = 8,
        epochs: int = 2,
        device: str | None = None,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        self.max_length = max_length
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def fit(self, texts: Iterable[str], labels: Iterable[int]) -> None:
        dataset = TextDataset(list(texts), list(labels), self.tokenizer, self.max_length)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        total_steps = len(loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )

        self.model.train()
        for _ in range(self.epochs):
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

    def predict(self, texts: Iterable[str]) -> List[BERTPrediction]:
        self.model.eval()
        predictions: List[BERTPrediction] = []
        with torch.no_grad():
            for text in texts:
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoding = {k: v.to(self.device) for k, v in encoding.items()}
                outputs = self.model(**encoding)
                probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)
                score = probs[1].item()
                label = int(score >= 0.5)
                predictions.append(BERTPrediction(label=label, score=score))
        return predictions
