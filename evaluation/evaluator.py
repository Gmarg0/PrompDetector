import torch
from torch.utils.data import DataLoader
from ..data.dataset import TwitterDataset
import time


def evaluate_model(model, tokenizer, test_df, config):
    test_dataset = TwitterDataset(test_df, tokenizer, config['training']['max_length'])
    test_loader = DataLoader(test_dataset, batch_size=config['evaluation']['batch_size'])

    model.eval()
    predictions = []
    labels = []
    total_time = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)

            start_time = time.time()
            outputs = model(input_ids, attention_mask=attention_mask)
            end_time = time.time()

            total_time += end_time - start_time

            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=1).cpu().tolist())
            labels.extend(batch['labels'].cpu().tolist())

    accuracy = sum(p == l for p, l in zip(predictions, labels)) / len(labels)
    avg_inference_time = total_time / len(test_loader)

    return {
        'accuracy': accuracy,
        'avg_inference_time': avg_inference_time
    }