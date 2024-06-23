import torch
from torch.utils.data import DataLoader
from data.dataset import TextClassificationDataset
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def evaluate_model(model, tokenizer, test_df, config):
    test_dataset = TextClassificationDataset(test_df, tokenizer, config['training']['max_length'])
    test_loader = DataLoader(test_dataset, batch_size=config['evaluation']['batch_size'])

    model.eval()
    all_predictions = []
    all_labels = []
    total_inference_time = 0
    num_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels']

            start_time = time.time()
            outputs = model(input_ids, attention_mask=attention_mask)
            end_time = time.time()

            total_inference_time += end_time - start_time
            num_samples += input_ids.size(0)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.tolist())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
    avg_inference_time = total_inference_time / num_samples

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'avg_inference_time': avg_inference_time
    }