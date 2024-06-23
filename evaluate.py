import yaml
import pandas as pd
import torch
from models.model_factory import get_model_and_tokenizer
from evaluation.evaluator import evaluate_model
from safetensors.torch import load_file


def main():
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    test_df = pd.read_csv(config['data']['test_path'])

    results = []

    for model_config in config['models']:
        model, tokenizer = get_model_and_tokenizer(
            model_config,
            config['training']['num_labels']
        )

        state_dict = load_file(f"./trained_models/{model_config['name']}/model.safetensors")
        model.load_state_dict(state_dict)

        model.to(model.device)

        evaluation_results = evaluate_model(model, tokenizer, test_df, config)
        results.append({
            'model': model_config['name'],
            **evaluation_results
        })

    for result in results:
        print(f"Model: {result['model']}")
        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"Precision: {result['precision']:.4f}")
        print(f"Recall: {result['recall']:.4f}")
        print(f"F1 Score: {result['f1_score']:.4f}")
        print(f"Average Inference Time: {result['avg_inference_time']:.4f} seconds")
        print()

if __name__ == '__main__':
    main()