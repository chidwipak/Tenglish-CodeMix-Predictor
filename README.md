# 🔮 Tenglish-CodeMix-Predictor: Telugu-English Code-Mixed Next Word Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.21+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)

**A next-word prediction system for Telugu-English code-mixed text using Statistical and Neural Language Models**

[Features](#-features) • [Demo](#-quick-demo) • [Installation](#-installation) • [Usage](#-usage) • [Results](#-results) • [Models](#-models)

</div>

---

> [!NOTE]
> **Project Origin**: This project was conceived and developed in **October 2025** as part of an academic NLP course. It was originally implemented and stored on a remote college server. It is now being pushed to GitHub for portfolio and internship application purposes.
>
> **Author**: Chidwipak Kuppani

---

## 🎯 Overview

This project addresses the challenge of predicting the next word in **Telugu-English code-mixed (Tenglish)** text — a unique linguistic phenomenon common in Indian digital communication where speakers fluently switch between languages within a single sentence.

```
Input:  "nenu eppudu"  (I always...)
Output: "untanu" ✓     (I will always be...)
```

For a deep dive into the implementation details, phases of development, and mathematical foundations, please read [PROJECT_DETAILS.md](PROJECT_DETAILS.md).

### What is Code-Mixing?
Code-mixing is the practice of alternating between two or more languages in a single conversation. For example:
> *"Nenu tomorrow office ki veltanu"* (I will go to office tomorrow)

This project builds models that understand and predict such mixed-language patterns using techniques ranging from N-grams to LSTMs implemented from scratch.

---

## ✨ Features

- 🧠 **4 Language Models**: N-gram (Kneser-Ney), HMM (Viterbi), Hybrid (Ensemble), and LSTM.
- 🔧 **Pure Python LSTM**: Neural network built **entirely from scratch using NumPy** (no PyTorch/TensorFlow) to demonstrate deep understanding of Backpropagation through Time (BPTT).
- 📊 **Complete Preprocessing Pipeline**: Custom tokenizer for mixed-script text.
- 🎮 **Interactive Demo**: Real-time prediction interface.
- 📈 **Comprehensive Evaluation**: Metrics including Top-k Accuracy and MRR.

---

## 🚀 Quick Demo

```bash
# Clone the repository
git clone https://github.com/chidwipaK/Tenglish-CodeMix-Predictor.git
cd Tenglish-CodeMix-Predictor

# Install dependencies
pip install numpy

# Run interactive demo
python interactive_demo.py
```

**Example Output:**
```
Enter text: nenu school ki

📊 Predictions:
┌─────────────┬────────────────────────────────┐
│ Model       │ Top Predictions                │
├─────────────┼────────────────────────────────┤
│ N-gram      │ veltanu, velta, veldam         │
│ HMM         │ velta, veltanu, potha          │
│ Hybrid      │ veltanu, velta, veldam         │
│ LSTM        │ veltanu, oka, untanu           │
└─────────────┴────────────────────────────────┘
```

---

## 📦 Installation

### Prerequisites
- Python 3.7 or higher
- pip

### Setup

```bash
# Clone repository
git clone https://github.com/chidwipaK/Tenglish-CodeMix-Predictor.git
cd Tenglish-CodeMix-Predictor

# Install dependencies
pip install numpy
```

---

## 💻 Usage

### 1. Interactive Testing (Recommended)
```bash
python interactive_demo.py
```

### 2. Train All Models
To retrain models from scratch (warning: LSTM training on CPU takes time):
```bash
python train_all_models.py
```

### 3. Evaluate Models
```bash
python run_final_evaluation.py
```

### 4. Use in Your Code
```python
from models.hybrid.hybrid_language_tagger import HybridLanguageTagger

# Load the best model (Hybrid)
model = HybridLanguageTagger()
model.load_models('models/hybrid_tagger.txt')

# Predict next word
context = ['nenu', 'eppudu']
predictions = model.predict_next_word(context, top_k=5)
print(predictions)
# Output: [('untanu', 0.51), ('unte', 0.28), ('avutanu', 0.11), ...]
```

---

## 📊 Results

### Model Comparison

| Model | Top-1 Accuracy | Top-5 Accuracy | MRR |
|:------|:-------------:|:--------------:|:---:|
| N-gram (10-gram) | 7.21% | 11.67% | 0.089 |
| HMM (4-state) | 4.72% | 10.27% | 0.067 |
| **Hybrid** ⭐ | **7.33%** | **12.40%** | **0.092** |
| LSTM (2 epochs) | 0.00% | 0.00% | 0.000 |

> 🏆 **Best Model**: Hybrid (N-gram + HMM combined)

### Why specific numbers?
- **Hybrid Superiority**: By combining the *structural* knowledge of HMM (language switching points) with the *contextual* memory of N-grams, the Hybrid model better captures the nuances of Tenglish.
- **LSTM Performance**: The LSTM model is a valid implementation but trained for only 2 epochs on a small dataset without GPU acceleration. It serves as a proof-of-concept for the architecture rather than a production model.

---

## 🧠 Models Implemented

### 1. N-gram Model
- **Type**: Statistical
- **Implementation**: 10-gram with **Kneser-Ney Smoothing**

### 2. Hidden Markov Model (HMM)
- **Type**: Statistical / Sequence Labeling
- **Algorithm**: **Viterbi Decoding** implemented from scratch.
- **Purpose**: To tag words as 'Telugu' or 'English' and find language transition probabilities.

### 3. Hybrid Model ⭐
- **Type**: Ensemble
- **Approach**: Weighted interpolation of N-gram and HMM probabilities based on confidence scores.

### 4. LSTM Model
- **Type**: Recurrent Neural Network
- **Architecture**: Embedding(128) -> LSTM(256) -> Dense(Vocab)
- **Implementation**: **Pure NumPy**. No Autograd. Gradients calculated manually.

---

## 🔮 Future Improvements

- [ ] Train LSTM with PyTorch/TensorFlow for GPU acceleration to reach convergence.
- [ ] Implement Transformer-based architecture (e.g., mBERT).
- [ ] Expand dataset to 10k+ sentences.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Made with ❤️ by [Chidwipak Kuppani](https://github.com/chidwipaK)**

</div>
