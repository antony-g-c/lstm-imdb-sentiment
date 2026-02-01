import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, classification_report

tf.random.set_seed(42)
np.random.seed(42)

num_words = 20000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

max_len = 300
x_train = pad_sequences(x_train, maxlen=max_len, padding="post", truncating="post")
x_test = pad_sequences(x_test, maxlen=max_len, padding="post", truncating="post")

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=num_words, output_dim=128, input_length=max_len),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, dropout=0.2)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=3,
    batch_size=128,
    verbose=1
)

proba = model.predict(x_test).ravel()
pred = (proba >= 0.5).astype(int)

print("\n=== LSTM Results ===")
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))
