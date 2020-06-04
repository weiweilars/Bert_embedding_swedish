import torch
import os
from transformers import BertTokenizer, BertModel
import numpy

DEFAULT_MODEL_PATH = "models/bert-base-cased"

class TextTransform():
    
    """Map the text to integers and dvice versa"""
    def __init__(self, pre_trained_model, max_len, do_lower_case=True):
        self.pre_trained_model = pre_trained_model
        self.max_len = max_len
        self.do_lower_case = do_lower_case
        
    def text_to_int(self, sentence):
        vocabulary = os.path.join(self.pre_trained_model, "vocab.txt")
        tokenizer = BertTokenizer(vocab_file=vocabulary, do_lower_case=self.do_lower_case)
        temp_token = []
        temp_token = ['[CLS]']+ tokenizer.tokenize(sentence)

        if len(temp_token) > self.max_len -1:
            temp_token = temp_token[:self.max_len-1]
            temp_token = temp_token + ['[SEP]']
        else:
            temp_token = temp_token + ['[SEP]']
            temp_token += ['[PAD]']*(self.max_len-len(temp_token))
        input_ids = tokenizer.convert_tokens_to_ids(temp_token)
        attention_masks = [int(i>0) for i in input_ids]
        
        return input_ids, attention_masks

class BertEmbeding_pt():
    """Make the Bert embedding"""
    def __init__(self, pre_trained_model=DEFAULT_MODEL_PATH, max_len=128, do_lower_case=True):
        self.pre_trained_model = pre_trained_model
        self.max_len = max_len
        self.do_lower_case=True
        self.text_transform = TextTransform(pre_trained_model, max_len)

    def embedding(self,text):
        tokens, mask = self.text_transform.text_to_int(text)
        model = BertModel.from_pretrained(self.pre_trained_model,output_hidden_states=False)

        model.eval()

        tokens_tensor = torch.tensor([tokens])
        mask_tensor = torch.tensor([mask])

        with torch.no_grad():
            outputs = model(tokens_tensor, mask_tensor)

        return(outputs[0].squeeze(0).numpy())
    
if __name__ == '__main__':

    text = "I am just trying, to do what I think it is right."
    
    my_embedding = BertEmbeding_pt()
    embedding_result = my_embedding.embedding(text)

    print(embedding_result)
