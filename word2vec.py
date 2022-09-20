import pandas as pd
import gensim
from gensim.models import Word2Vec

import time

df = pd.read_csv('./Datasets/20191226-reviews.csv')

print(df.head())

print(df.shape)

print("Before preprocessing:")
print(df['body'][0])

print("After preprocessing:")
print(gensim.utils.simple_preprocess(df['body'][0]))

t = time.process_time()

df['body'] = df['body'].fillna('').astype(str)

df['review_text'] = df['body'].apply(gensim.utils.simple_preprocess)

elapsed_time = time.process_time() - t

print(f"Time to preprocess the data: {elapsed_time}")

model = gensim.models.Word2Vec(
    window=10,
    min_count=2,
    workers=4
)

model.build_vocab(df['review_text'], progress_per=1000)

print(f"Total number of sentences in the corpus = {model.corpus_count}")
model_file_name = "./word2vec-cellphone-review.model"

t = time.process_time()

model.train(df['review_text'], total_examples=model.corpus_count, epochs=10)

elapsed_time = time.process_time() - t

print(f"Time to train the Word2Vec model: {elapsed_time}")

model.save(model_file_name)

model = Word2Vec.load(model_file_name)
print("Most similar words to 'bad'")
print(model.wv.most_similar("bad"))

print(f"Cosine (similarity) score between 'cheap' and 'inexpensive' = {model.wv.similarity('cheap', 'inexpensive')}")

print(f"Cosine (similarity) score between 'cheap' and 'phone' = {model.wv.similarity('cheap', 'phone')}")

