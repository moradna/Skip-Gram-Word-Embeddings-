# 🧠 Skip-Gram Word Embeddings (with Negative Sampling)

This project implements a basic **Skip-Gram model** for learning word embeddings using **negative sampling**.  
The model learns from a corpus of English sentences and creates vector representations of words based on their context.

---

## 🚀 Main Features

- Implements Skip-Gram from scratch using NumPy
- Supports negative sampling
- Includes early stopping and loss tracking
- Provides utilities for:
  - Cosine similarity
  - Finding most similar words
  - Word analogy tasks (e.g., king - man + woman ≈ queen)

---

## ✨ Example Usage

```python
from SkipGram import SkipGram, normalize_text

sentences = normalize_text("data/corpus.txt")
sg = SkipGram(sentences)
sg.learn_embeddings()
print(sg.get_closest_words("king"))

---
## 📁 Files

- `SkipGram.py` – Main implementation of the Skip-Gram model with negative sampling, training logic, and embedding functions.
- `Tester.py` – Script for testing functionality: computing similarities, finding analogies, etc.
- `corpus.txt` – Text file containing the raw corpus used for training the model.
- `model.pkl` – Saved model (via pickle), created after training for future use (can be loaded using `load_model()`).
