import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import time

# create a variable for downloading the BERT model
encoder_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4'

# for the BERT model there is a preprocessing URL, which will preprocess the text
preprocess_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'

# create a hub layer, similar to a function pointer
bert_preprocess_model = hub.KerasLayer(preprocess_url)

test_text = ['very good food', 'we love deep learning']

t = time.process_time()
text_preprocessed = bert_preprocess_model(test_text)

elapsed_time = time.process_time() - t
print(f"Time to preprocess the text: {elapsed_time}")

# preprocessing will produce a dictionary object:
print(f"The dictionary of the pre-processing result: {text_preprocessed.keys()}")

# the following matrices are the input mask. With this model 128 words is the maximum length of the sentence.
# BERT uses special characters at the beginning of the input (CLS) and at the end of the input (SEP)
print(text_preprocessed['input_mask'])

# now creating the function pointer to the model:
bert_model = hub.KerasLayer(encoder_url)

t = time.process_time()

bert_features = bert_model(text_preprocessed)

elapsed_time = time.process_time() - t
print(f"Time to pass through the model: {elapsed_time}")

print(f"The dictionary of the bert features: {bert_features.keys()}")

# the 'pooled_output' is an embedding for the entire sentence, typically of size 768
print(f"Pooled_outputs (Embeddings for the entire sentence): {bert_features['pooled_output']}")

# the 'sequence_output' is an embedding for the individual words, typically a matrix of size 128x768 for each word
print(f"Pooled_outputs (Embeddings for the individual words): {bert_features['sequence_output']}")

# the 'encoder_outputs' is a list of 12 tensors, each of size 128x768. Number 12 is the nr. of the encoders of BERT
print(f"Encoder_outputs (Embeddings for the entire sentence): {bert_features['encoder_outputs']}")

# Dataset from https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset/code
df = pd.read_csv('./Datasets/spam.csv', encoding='latin-1')

print(df.groupby('v1').describe())

df_spam = df[df['v1'] == 'spam']
df_ham = df[df['v1'] == 'ham']

df_ham = df_ham.sample(df_spam.shape[0])

df_balanced = pd.concat([df_spam, df_ham])

print(df_balanced['v1'].value_counts())

df_balanced['spam'] = df_balanced['v1'].apply(lambda x: 1 if x == 'spam' else 0)

X_train, X_test, y_train, y_test = train_test_split(df_balanced['v2'], df_balanced['spam'], stratify=df_balanced['spam'])

print(X_train.head())


def get_sentence_embedding(sentences):
    preprocessed = bert_preprocess_model(sentences)
    return bert_model(preprocessed)['pooled_output']


def build_model():
    # Define the BERT layers
    input = Input(shape=(), dtype=tf.string, name="text")
    preprocessed_text = bert_preprocess_model(input)
    embeddings = bert_model(preprocessed_text)

    # Neural Network layers:
    drop = Dropout(0.1, name='dropout')(embeddings['pooled_output'])
    output = Dense(1, activation='sigmoid', name='output')(drop)

    # Construct the final model:
    model = Model(inputs=[input], outputs=[output])

    metrics = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=metrics)

    return model


model = build_model()

t = time.process_time()

model.fit(X_train, y_train, epochs=10)

elapsed_time = time.process_time() - t
print(f"Time to train the Keras model: {elapsed_time}")

model.save("./models/1/")
model.save("./models/2/")
model.save("./models/3/")

model.evaluate(X_test, y_test)

y_pred = model.predict(X_test)
y_pred = y_pred.flatten()

y_pred = np.where(y_pred > 0.5, 1, 0)

cm = confusion_matrix(y_test, y_pred)

print(cm)

print(classification_report(y_test, y_pred))

# instructions for Docker and TF Serving:
# docker pull tensorflow/serving
# docker run -it -v C:\WORK\Projects\Deep-Learning-NLP\models: /tf_serving - p 8605: 8605 --entrypoint /bin/bash tensorflow/serving
# tensorflow_model_server --rest_api_port=8605 --model_name=email_model --model_base_path=/tf_serving/


