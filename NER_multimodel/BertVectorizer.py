import torch
from transformers import BertTokenizer, BertModel

class BertVectorizer:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def vectorize(self, text):
        # Tokenize the input text
        tokens = self.tokenizer.tokenize(text)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # Add special tokens and convert to tensor
        input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(self.device)

        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(input_ids)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()

        return embeddings
bert_vectorizer = BertVectorizer()
text = "This is a sample sentence."
vector = bert_vectorizer.vectorize(text)
print(vector)
