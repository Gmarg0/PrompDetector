from transformers import AutoTokenizer, AutoModelForSequenceClassification

def get_model_and_tokenizer(model_config, num_labels):
    model_name = model_config['name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model, tokenizer