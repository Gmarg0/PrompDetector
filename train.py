import yaml
from data.dataset import load_data
from models.model_factory import get_model_and_tokenizer
from training.trainer import train_model
from sklearn.model_selection import train_test_split

def main():
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    train_df, test_df = load_data(config)
    train_df, eval_df = train_test_split(train_df, test_size=0.1)

    for model_config in config['models']:
        model, tokenizer = get_model_and_tokenizer(
            model_config,
            config['training']['num_labels']
        )
        train_model(model, tokenizer, train_df, eval_df, config)

if __name__ == '__main__':
    main()