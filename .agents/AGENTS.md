# Project Scope: Comment Toxicity Classification

This file contains rules and guidelines for AI coding assistants working in this repository.

## Development Rules
1. **Coding Standard**: All Python code MUST follow the PEP-8 standard. Use clear variable names and document logic in docstrings where necessary.
2. **Modular Architecture**: Code should be organized into functional blocks (e.g., separating UI, inference, and preprocessing).
3. **Environment**: We are building a deep learning model for toxicity classification. Training is meant to be done externally (e.g., Google Colab), so local code focuses on preprocessing, inference, and Streamlit application deployment.
4. **Library Preferences**: 
   - Model: TensorFlow/Keras (BiLSTM).
   - UI: Streamlit.
   - Data Manipulation: Pandas, NumPy.
5. **Git Workflows**: Ensure frequent commits after meaningful logic changes. Do not commit large datasets or trained model weight files (`.h5`, `.keras`, `.pkl`) to version control.
