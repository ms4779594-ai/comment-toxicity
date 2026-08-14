# Comment Toxicity Detection System

An automated system capable of detecting and flagging toxic comments in real-time, built with TensorFlow, Keras, and Streamlit. This project uses a deep learning approach (Bidirectional LSTM) to identify various forms of toxicity in online discussions.

## Project Structure
- `app.py`: Streamlit application serving as the user interface.
- `colab_training.py`: Script to train the BiLSTM model on Google Colab (to leverage GPU).
- `src/inference.py`: Modular component to handle model loading and real-time predictions.
- `requirements.txt`: Python package dependencies.
- `.agents/AGENTS.md`: AI development guidelines and project rules.

## Quick Start & Setup

### 1. Training the Model (Google Colab)
Since this model requires significant resources to train, we recommend using Google Colab's free GPU:
1. Open [Google Colab](https://colab.research.google.com/) and create a new notebook.
2. Go to **Runtime** > **Change runtime type** > Select **GPU**.
3. Upload `train.csv` (and `test.csv` if needed) to the Colab environment.
4. Copy the contents of `colab_training.py` into a cell and run it.
5. Once complete, download the generated files: `toxicity_model.h5` and `vectorizer.pkl`.
6. Place these two files in this local repository root (`Project-2/`).

### 2. Running the Application Locally
After acquiring the model files:
```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## Features
- **Real-Time Analysis**: Enter a comment and instantly get a breakdown of its toxicity across 6 categories (Toxic, Severe Toxic, Obscene, Threat, Insult, Identity Hate).
- **Bulk CSV Analysis**: Upload a CSV with a `comment_text` column, and the app will generate a new file with predicted scores appended for easy downloading.
- **Responsive UI**: A styled dashboard displaying dynamic warning colors based on prediction confidence.

## Demonstration
LinkedIn: https://lnkd.in/p/gqtvHr_q