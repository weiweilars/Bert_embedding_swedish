from bert_embedding_pt import BertEmbeding_pt

text = "This is just the test text."

embedding_result = BertEmbeding_pt().embedding(text)

print(embedding_result)

