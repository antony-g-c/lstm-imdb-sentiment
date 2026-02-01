# LSTM IMDB Sentiment Classifier

This project trains an LSTM model to predict whether a movie review is **positive** or **negative**.

## What this shows
- Understanding of **sequence data** (text)
- Using **Embeddings** to represent words
- Using an **LSTM** to learn patterns in word order
- Evaluating a classifier using accuracy and a classification report

## Dataset
Uses the built-in IMDB dataset from `tensorflow.keras.datasets.imdb`.

## How it works (simple)
1. Load IMDB reviews (as token numbers)
2. Pad reviews so they are the same length
3. Train an Embedding + BiLSTM model
4. Predict sentiment on the test set

## How to run
```bash
pip install -r requirements.txt
python train_lstm.py