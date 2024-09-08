from transformers import RobertaTokenizer, RobertaForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score
from datasets import Dataset

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)

    return {"accuracy": acc}


def plot_accuracy(log_history, save_path="accuracy_plot.png"):
    unique_epochs = []
    unique_train_acc = []

    for entry in log_history:
        epoch = entry.get('epoch')
        accuracy = entry.get('eval_accuracy')
        
        if epoch and accuracy and epoch not in unique_epochs:
            unique_epochs.append(epoch)
            unique_train_acc.append(accuracy)
    
    plt.plot(unique_epochs, unique_train_acc, label='Validation Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy over Epochs on Validation Set")

    plt.xticks(ticks=range(int(min(unique_epochs)), int(max(unique_epochs)) + 1)) 
    
    plt.savefig(save_path) 


def main():
    train_df = pd.read_csv("./data/train.tsv", sep="\t", header=None, names=["label", "sentence"])
    test_df = pd.read_csv("./data/test.tsv", sep="\t", header=None, names=["label", "sentence"])
    dev_df = pd.read_csv("./data/dev.tsv", sep="\t", header=None, names=["label", "sentence"])
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    dev_dataset = Dataset.from_pandas(dev_df)

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    def tokenize_function(dataset):
        return tokenizer(dataset['sentence'], padding="max_length", truncation=True)

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)
    tokenized_dev_dataset = dev_dataset.map(tokenize_function, batched=True)

    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

    training_args = TrainingArguments(
        output_dir="./original/results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=True,
        logging_dir='./original/logs',  
        logging_steps=10
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_dev_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    eval_results = trainer.evaluate(eval_dataset=tokenized_dev_dataset)
    print(f"Validation results: {eval_results}")

    test_results = trainer.evaluate(eval_dataset=tokenized_test_dataset)
    print(f"Test results: {test_results}")

    log_history = trainer.state.log_history
    plot_accuracy(log_history, save_path="./original/accuracy_plot.png") 


if __name__ == "__main__":
    main()