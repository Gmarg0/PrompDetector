# PrompDetector

PrompDetector is a PyTorch-based project for training and evaluating text classification models. While the primary use case demonstrated here is hate speech detection, this framework is flexible and can be adapted for any binary or multi-label text classification task.

## Project Structure

- **data**: Contains the dataset loading and preprocessing scripts.
- **models**: Includes the model factory for loading pre-trained models and tokenizers.
- **training**: Scripts for training models.
- **evaluation**: Scripts for evaluating trained models.
- **config**: Configuration files for specifying model and training parameters.

## Features

1. **Dataset Handling**: A custom `TextClassificationDataset` class to preprocess text data using a tokenizer and prepare it for model training and evaluation.
2. **Model Training**: Scripts to train models on the provided dataset.
3. **Model Evaluation**: Scripts to evaluate the performance of trained models using metrics such as accuracy, precision, recall, F1 score, and average inference time.
4. **Configuration Driven**: Easy to configure through YAML files for different experiments.

## Installation

1. Clone the repository:
   bash
   git clone https://github.com/Gmarg0/PrompDetector.git
   cd PrompDetector
2. Install the required packages:
pip install -r requirements.txt
Configuration
The configuration file config/config.yaml contains all necessary parameters for data paths, model specifications, and training configurations.

## Usage
### Training
To train the model, run the following command:

python training/train.py
This script will:

1. Load the training data.
2. Split the data into training and evaluation sets.
3. Train the specified models using the training set.
4. Save the trained models.


### Evaluation
To evaluate the model, run:

python evaluation/evaluate.py
This script will:

1. Load the test data.
2. Load the trained models.
3. Evaluate the models on the test set.
4. Print the evaluation results, including accuracy, precision, recall, F1 score, and average inference time.


## Models
By default, this project compares the following two models:

BERTHateSpeech - https://huggingface.co/GalMargo/BERTHateSpeech
DistiBERTHateSpeech - https://huggingface.co/GalMargo/DistiBERTHateSpeech
These models are fine-tuned for hate speech detection but can be adapted for other text classification tasks.

## Example Configuration
An example configuration can be found in config_file (config/config.yaml):

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.