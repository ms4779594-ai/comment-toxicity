"""
Toxicity Model Training Script for Google Colab

Instructions:
1. Open Google Colab (https://colab.research.google.com/).
2. Create a new notebook.
3. Change Runtime type to 'GPU' (Runtime -> Change runtime type -> Hardware accelerator -> GPU).
4. Upload `train.csv` to the Colab environment (or mount your Google Drive if it's there).
5. Copy and paste the code below into a cell and run it.
6. Once training completes, download `toxicity_model.h5` and `vectorizer.pkl` back to your local `Project-2` folder.
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.models import Sequential
import pickle

def main():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    
    # 1. Load Data
    print("Loading dataset...")
    df = pd.read_csv('train.csv')
    
    # Features and labels
    X = df['comment_text']
    y = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values

    # 2. Text Vectorization
    print("Initializing TextVectorization...")
    MAX_FEATURES = 200000
    vectorizer = TextVectorization(max_tokens=MAX_FEATURES,
                                   output_sequence_length=1800,
                                   output_mode='int')
    
    print("Adapting vectorizer to text (this may take a few minutes)...")
    vectorizer.adapt(X.values)
    vectorized_text = vectorizer(X.values)

    # Save vectorizer configuration and vocabulary for Streamlit inference later
    # Note: TextVectorization layer in Keras 3 doesn't return vocabulary in get_weights()
    print("Saving vectorizer config and vocabulary...")
    vectorizer_data = {
        'config': vectorizer.get_config(),
        'vocabulary': vectorizer.get_vocabulary()
    }
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer_data, f)

    # 3. Create TensorFlow Dataset
    dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))
    dataset = dataset.cache().shuffle(160000).batch(16).prefetch(8)
    
    train_size = int(len(dataset) * 0.7)
    val_size = int(len(dataset) * 0.2)
    test_size = int(len(dataset) * 0.1)
    
    train = dataset.take(train_size)
    val = dataset.skip(train_size).take(val_size)
    test = dataset.skip(train_size + val_size).take(test_size)

    # 4. Build Model
    print("Building BiLSTM model...")
    model = Sequential([
        Embedding(MAX_FEATURES + 1, 32),
        Bidirectional(LSTM(32, activation='tanh')),
        Dense(128, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(6, activation='sigmoid') # 6 outputs for our 6 classes
    ])
    
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
    model.summary()
    
    # 5. Train Model
    print("Starting training...")
    history = model.fit(train, epochs=1, validation_data=val) # Set epochs=5 for better performance, using 1 for speed testing
    
    # 6. Save Model
    print("Saving model...")
    model.save('toxicity_model.h5')
    print("Training complete! Download 'toxicity_model.h5' and 'vectorizer.pkl'")

if __name__ == '__main__':
    main()
