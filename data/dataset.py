import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

class TwitterDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.labels = dataframe['label'].tolist()
        self.texts = [
            tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ) for text in dataframe['text']
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            'input_ids': self.texts[idx]['input_ids'].squeeze(0),
            'attention_mask': self.texts[idx]['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }
        return item

def load_data(config):
    df = pd.read_csv(config['data']['train_path'])
    df = df[['tweet', 'class']].rename(columns={'tweet': 'text', 'class': 'label'})
    train_df, test_df = train_test_split(
        df,
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )
    test_df.to_csv(config['data']['test_path'], index=False)
    return train_df, test_df