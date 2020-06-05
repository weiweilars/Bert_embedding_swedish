import os
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
import numpy as np

DEFAULT_MODEL_PATH = "models/bert-base-uncased"
if not os.path.exists(DEFAULT_MODEL_PATH):
    DEFAULT_MODEL_PATH = "af-ai-center/bert-base-swedish-uncased"

class TextTransform():
    
    """Map the text to integers and dvice versa"""
    def __init__(self, pre_trained_model, max_len, do_lower_case=True):
        self.pre_trained_model = pre_trained_model
        self.max_len = max_len
        self.do_lower_case = do_lower_case
        
    def text_to_int(self, sentence):
        tokenizer = BertTokenizer.from_pretrained(self.pre_trained_model, do_lower_case=self.do_lower_case)
        temp_token = []
        temp_token = ['[CLS]']+ tokenizer.tokenize(sentence)

        if len(temp_token) > self.max_len -1:
            temp_token = temp_token[:self.max_len-1]
            temp_token = temp_token + ['[SEP]']
        else:
            temp_token = temp_token + ['[SEP]']
            temp_token += ['[PAD]']*(self.max_len-len(temp_token))

        #print(temp_token)
        input_ids = tokenizer.convert_tokens_to_ids(temp_token)
        #print(input_ids)
        attention_masks = [int(i>0) for i in input_ids]
        #print(attention_masks)

        segment_ids = [0] * len(input_ids)
        
        return input_ids, attention_masks, segment_ids

class BertEmbeding_tf():
    """Make the Bert embedding"""
    def __init__(self, pre_trained_model=DEFAULT_MODEL_PATH, max_len=128, do_lower_case=True):
        self.pre_trained_model = pre_trained_model
        self.max_len = max_len
        self.do_lower_case=True
        self.text_transform = TextTransform(pre_trained_model, max_len)

    def embedding(self,text):
        tokens, masks, seg = self.text_transform.text_to_int(text)
        input_ids = tf.keras.layers.Input(shape=(self.max_len,), dtype=tf.int32,
                                       name="input_ids")
        attention_mask = tf.keras.layers.Input(shape=(self.max_len,), dtype=tf.int32,
                                   name="attention_mask")
        token_type_ids  = tf.keras.layers.Input(shape=(self.max_len,), dtype=tf.int32,
                                    name="token_type_ids") 
        bert_layer = TFBertModel.from_pretrained(self.pre_trained_model)

        outputs = bert_layer({'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask' : attention_mask})

        model = tf.keras.Model(inputs={'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask' : attention_mask}, outputs = outputs)

        input_token = tf.constant(tokens)[None, :]
        token_input = tf.constant(seg)[None, :]
        mask_input = tf.constant(masks)[None, :]

        input = {'input_ids':input_token, 'token_type_ids': token_input, 'attention_mask': mask_input}
        output = model(input)
        return(np.asarray(tf.squeeze(output[0])))
        

if __name__ == '__main__':

    text = "I am just trying, to do what I think it is right."

    my_embedding = BertEmbeding_tf()

    embedding_result = my_embedding.embedding(text)

    print(embedding_result)

