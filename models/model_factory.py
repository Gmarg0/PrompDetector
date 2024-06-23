from transformers import (
    BertTokenizer, BertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification,
    DistilBertTokenizer, DistilBertForSequenceClassification
)

MODEL_CONFIG = {
    'bert': (BertTokenizer, BertForSequenceClassification),
    'roberta': (RobertaTokenizer, RobertaForSequenceClassification),
    'distilbert': (DistilBertTokenizer, DistilBertForSequenceClassification)
}

def get_model_and_tokenizer(model_name, model_type, num_labels):
    tokenizer_cls, model_cls = MODEL_CONFIG[model_type]
    tokenizer = tokenizer_cls.from_pretrained(model_name)
    model = model_cls.from_pretrained(model_name, num_labels=num_labels)
    return model, tokenizer