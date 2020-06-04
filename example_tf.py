from bert_embedding_tf import BertEmbeding_tf

text = "This is just the test text."

embedding_result = BertEmbeding_tf().embedding(text)

print(embedding_result)

