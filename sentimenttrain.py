# this file is note using.....
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
DATA_ROOT = "models"
NUM_EPOCHS = 1
CLS_MODEL_SAVE_PATH = f"{DATA_ROOT}/classification/"
PRETRAINED_MODEL = "distilbert-base-uncased"

class ClassificationTrainDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class ClassificationTestDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

def train_model(dataframe):
    data = pd.read_csv(dataframe.file, index_col=0)
    data.columns = ["text", "label"]

    x = data["text"].values
    y = data["label"].values

    # Split into training and validation sets
    train_data, val_data, train_labels, val_labels = train_test_split(y,x)
    # Load pre-trained AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    
    # Tokenize
    train_tokens = tokenizer(
        list(train_data),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )

    val_tokens = tokenizer(
        list(val_data),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )

    train_dataset = ClassificationTrainDataset(train_tokens, train_labels)
    val_dataset = ClassificationTrainDataset(val_tokens, val_labels)

    # Train the model
    training_args = TrainingArguments(
        output_dir=CLS_MODEL_SAVE_PATH,  # output directory
        num_train_epochs=NUM_EPOCHS,
        evaluation_strategy="epoch",  # total number of training epochs
    )

    model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL)

    # Trainer object
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

    trainer.save_model(CLS_MODEL_SAVE_PATH)

    return


# df = pd.read_csv('train.csv')
# asd = train_model("train.csv")
# print(asd)

from fastapi import FastAPI, File, UploadFile

app = FastAPI()


@app.post('/train/')
async def train(dataframe: UploadFile = File(...)):
    train_model(dataframe)
    return {"Message": "Model training has started!"}