import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
from src.data.dataloader import NewsDataset

def evaluate_model(model_path, dataset_path, batch_size=8, max_length=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.to(device)

    dataset = NewsDataset(dataset_path, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == batch['labels']).sum().item()
            total += batch['labels'].size(0)

    accuracy = correct / total
    print(f'Accuracy: {accuracy:.4f}')

if __name__ == "__main__":
    evaluate_model('output/', 'data/processed_articles.json', batch_size=8, max_length=512)

