import torch
from torch.utils.data import DataLoader
from transformers import XLNetTokenizer
from datasets import load_dataset
import pandas as pd

from src.model import BiLSTM
from src.datasets import IMDBDataset, DelhiMetroDataset
from src.train import train, evaluate
from utils import plot_confusion_matrix
from src.preprocess import preprocess_text

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")

# ----------------------------- IMDb Data ---------------------------------
print("Loading IMDb dataset...")
imdb_dataset = load_dataset("imdb")
imdb_dataset["train"] = [{"text": preprocess_text(example["text"]), "label": example["label"]} for example in imdb_dataset["train"]]
imdb_dataset["test"] = [{"text": preprocess_text(example["text"]), "label": example["label"]} for example in imdb_dataset["test"]]

train_dataset = IMDBDataset(imdb_dataset["train"], tokenizer, max_len=256)
test_dataset = IMDBDataset(imdb_dataset["test"], tokenizer, max_len=256)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Load Model
model = BiLSTM().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

# Train and Evaluate on IMDb
for epoch in range(3):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    print(f"Epoch {epoch+1}, IMDb Loss: {train_loss:.4f}")

# Evaluate
imdb_report, imdb_accuracy = evaluate(model, test_loader, device)
print("IMDb Classification Report:\n", imdb_report)
print(f"IMDb Accuracy: {imdb_accuracy:.4f}")

# ----------------------- Delhi Metro Data ---------------------------------
print("Loading Delhi Metro dataset...")
delhi_metro_data = pd.read_csv('data/delhi_metro.csv')
delhi_metro_data['Sentiment'] = delhi_metro_data['Sentiment'].map({'positive': 1, 'negative': 0, 'positiv': 1})

# Preprocess comments
delhi_metro_data['Cleaned_Comment'] = delhi_metro_data['Comment'].apply(preprocess_text)

# Split into train and test
train_df = delhi_metro_data.sample(frac=0.8, random_state=42)
test_df = delhi_metro_data.drop(train_df.index)

train_dataset = DelhiMetroDataset(train_df, tokenizer, max_len=256)
test_dataset = DelhiMetroDataset(test_df, tokenizer, max_len=256)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Train and Evaluate on Delhi Metro
for epoch in range(3):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    print(f"Epoch {epoch+1}, Delhi Metro Loss: {train_loss:.4f}")

# Evaluate
delhi_metro_report, delhi_metro_accuracy = evaluate(model, test_loader, device)
print("Delhi Metro Classification Report:\n", delhi_metro_report)
print(f"Delhi Metro Accuracy: {delhi_metro_accuracy:.4f}")

# Plot Confusion Matrix for Delhi Metro
_, predicted_labels = evaluate(model, test_loader, device)
plot_confusion_matrix(test_df['Sentiment'], predicted_labels, "Delhi Metro")

