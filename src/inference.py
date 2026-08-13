import os
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TextVectorization

class ToxicityPredictor:
    def __init__(self, model_path='toxicity_model.h5', vectorizer_path='vectorizer.pkl'):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.model = None
        self.vectorizer = None
        self.labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    def load(self):
        """Loads the trained model and vectorizer from disk."""
        if not os.path.exists(self.model_path) or not os.path.exists(self.vectorizer_path):
            return False
            
        try:
            # Load the model
            self.model = load_model(self.model_path)
            
            # Load the vectorizer config and vocabulary
            with open(self.vectorizer_path, 'rb') as f:
                vectorizer_data = pickle.load(f)
                
            self.vectorizer = TextVectorization.from_config(vectorizer_data['config'])
            # Set the vocabulary directly to initialize the lookup table
            self.vectorizer.set_vocabulary(vectorizer_data['vocabulary'])
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def predict(self, text):
        """Predicts toxicity levels for a given text."""
        if not self.model or not self.vectorizer:
            raise ValueError("Model or vectorizer not loaded.")
            
        vectorized_text = self.vectorizer([text])
        prediction = self.model.predict(vectorized_text, verbose=0)[0]
        
        results = {label: float(pred) for label, pred in zip(self.labels, prediction)}
        return results
