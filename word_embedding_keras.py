import numpy as np

from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding

reviews = ['amazing car',
           'nice laptop',
           'will go again',
           'excellent product',
           'very good restaurant',
           'horrible food',
           'bad person',
           'low quality',
           'needs improvement',
           'never go there']

sentiment_truth = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

vocab_size = 40

print(one_hot("amazing car", vocab_size))  # 40 is the vocabulary size, so one_hot will give a number from 0 to 39 for each word

encoded_reviews = [one_hot(rev, vocab_size) for rev in reviews]

print(encoded_reviews)

max_sentence_length = 3

padded_reviews = np.array(pad_sequences(encoded_reviews, maxlen=max_sentence_length, padding='post'))
print(padded_reviews)

embeded_vector_size = 4

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embeded_vector_size,
                    input_length=max_sentence_length, name="embedding"))

model.add(Flatten())
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

history = model.fit(padded_reviews, sentiment_truth, epochs=50, verbose=1)

loss, accuracy = model.evaluate(padded_reviews, sentiment_truth)
print(f"Accuracy={accuracy}")

# Word embeddings are just the parameters inside the neural network
weights = model.get_layer("embedding").get_weights()[0]
print(weights)

