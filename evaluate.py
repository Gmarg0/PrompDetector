import yaml
import pandas as pd
from src.models.model_factory import get_model_and_tokenizer
from src.evaluation.evaluator import evaluate_model


def main():
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    test_df = pd.read_csv(config['data']['test_path'])

    results = []

    for model_config in config['models']:
        model, tokenizer = get_model_and_tokenizer(
            model_config['name'],
            model_config['type'],
            config['training']['num_labels']
        )
        model.load_state_dict(torch.load(f"./model_{model_config['name']}/pytorch_model.bin"))

        evaluation_results = evaluate_model(model, tokenizer, test_df, config)
        results.append({
            'model': model_config['name'],
            **evaluation_results
        })

    for result in results:
        print(f"Model: {result['model']}")
        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"Average Inference Time: {result['avg_inference_time']:.4f} seconds")
        print()


if __name__ == '__main__':
    main()