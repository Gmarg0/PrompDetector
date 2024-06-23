from transformers import Trainer, TrainingArguments
from data.dataset import TextClassificationDataset

def train_model(model, tokenizer, train_df, eval_df, config):
    train_dataset = TextClassificationDataset(train_df, tokenizer, config['training']['max_length'])
    eval_dataset = TextClassificationDataset(eval_df, tokenizer, config['training']['max_length'])

    training_args = TrainingArguments(
        output_dir=f"./model_{model.name_or_path}",
        num_train_epochs=config['training']['num_train_epochs'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        logging_dir=f"./logs_{model.name_or_path}",
        logging_steps=config['training']['logging_steps'],
        evaluation_strategy=config['training']['eval_strategy'],
        save_strategy=config['training']['save_strategy']
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    trainer.train()
    model.save_pretrained(f"./model_{model.name_or_path}")