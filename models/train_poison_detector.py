from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune DeBERTa for binary poison classification.")
    parser.add_argument("--data", default="data/processed/poison_training.csv")
    parser.add_argument("--base-model", default="microsoft/deberta-v3-base")
    parser.add_argument("--output", default="models/checkpoints/poison-detector")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    from datasets import Dataset
    import pandas as pd
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

    df = pd.read_csv(args.data)
    dataset = Dataset.from_pandas(df[["text", "label"]], preserve_index=False)
    split = dataset.train_test_split(test_size=0.2, seed=42)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

    split = split.map(tokenize, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=2)
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=split["train"], eval_dataset=split["test"], tokenizer=tokenizer)
    trainer.train()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Saved poison detector to {args.output}")


if __name__ == "__main__":
    main()
