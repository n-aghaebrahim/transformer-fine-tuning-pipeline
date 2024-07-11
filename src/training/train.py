import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler
from tqdm import tqdm
from src.data.dataloader import NewsDataset
from src.models.model import get_model_and_tokenizer

def train_model(model_name, dataset_path, output_dir, num_epochs=3, batch_size=8, lr=5e-5, max_length=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = get_model_and_tokenizer(model_name)
    model.to(device)

    dataset = NewsDataset(dataset_path, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = num_epochs * len(dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    model.train()
    for epoch in range(num_epochs):
        for batch in tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    train_model('distilbert-base-uncased', 'data/processed_articles.json', 'output/', num_epochs=3, batch_size=8, lr=5e-5, max_length=512)

