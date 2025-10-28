# 🔮 Tenglish Code-Mixed Next Word Predictor: End-to-End Implementation Details

**Author**: Chidwipak Kuppani
**Date**: October 2025 (Project Completion)
**Status**: Completed (Research & Implementation)

---

## 1. Project Concept & Motivation

### The Problem
In the Indian subcontinent, it is extremely common for speakers to mix English with their native language (e.g., Telugu) in daily digital communication (WhatsApp, Social Media). This phenomenon is called **Code-Mixing**.
Standard NLP models (like GPT-3 or BERT trained on pure English) fail to understand or predict this mixed text accurately.
Example: *"Nenu tomorrow office ki veltanu"* (I will go to office tomorrow).

### The Solution
This project builds a **Language Model (LM) specifically for Telugu-English Code-Mixed text (Tenglish)** to predict the next word in a sentence. It explores the evolution of NLP approaches from statistical methods to neural networks, implemented entirely from scratch to demonstrate fundamental understanding.

---

## 2. Technical Stack

*   **Language**: Python 3.7+
*   **Core Logic**: NumPy (for all matrix operations, vectorization)
*   **No High-Level Frameworks**: The LSTM and HMM models are built using **pure Python and NumPy** without PyTorch or TensorFlow. This was a deliberate choice to demonstrate deep mathematical understanding of Forward Propagation, Backpropagation through Time (BPTT), and Viterbi Decoding during the academic research phase.
*   **Data Processing**: Regular Expressions (Regex), Collections

---

## 3. Phase-Wise Development Journey

This project was executed in 6 distinct phases over the course of October 2025.

### Phase 1: Data Acquisition & Preprocessing
**Objective**: Prepare raw social media text for modeling.
*   **Data Source**: Collected ~2,400 sentences of casual Telugu-English conversations.
*   **Preprocessing Pipeline**:
    *   **Tokenization**: Custom regex-based tokenizer to handle mixed scripts and punctuation.
    *   **Normalization**: Lowercasing, removing special characters often found in chat text.
    *   **Data Split**: 70% Train, 15% Validation, 15% Test.

### Phase 2: Statistical Modeling (N-gram)
**Objective**: Establish a baseline using classical NLP.
*   **Model**: **10-gram Model** with Backoff.
*   **Innovation**: Implemented **Kneser-Ney Smoothing** logic manually to handle unseen n-grams (zero probability problem).
*   **Outcome**: Good at memorizing common phrases but failed significantly on novel contexts due to sparsity.

### Phase 3: Language Tagging (HMM)
**Objective**: Understand the *structure* of code-mixing.
*   **Model**: **Hidden Markov Model (HMM)**.
*   **Implementation**:
    *   **Emission Probabilities**: $P(Word | Language)$
    *   **Transition Probabilities**: $P(Language_{t} | Language_{t-1})$
    *   **Algorithm**: Implemented **Viterbi Algorithm** from scratch to find the most likely sequence of language tags (Telugu vs. English) for a given sentence.
*   **Insight**: Code-mixing isn't random; it follows grammatical rules (Matrix Language Frame Theory).

### Phase 4: The Hybrid Approach (Ensemble)
**Objective**: Combine structure (HMM) with context (N-gram).
*   **Idea**: Use HMM to predict the *language* of the next word, and use that probability to re-rank specific candidates suggested by the N-gram model.
*   **Mechanism**: A weighted interpolation of $P_{ngram}(w|context)$ and $P_{hmm}(w|tag)$.
*   **Result**: Outperformed both standalone models, achieving the highest accuracy (~12.4% Top-5).

### Phase 5: Neural Modeling (LSTM)
**Objective**: Capture long-range dependencies.
*   **Model**: **Long Short-Term Memory (LSTM)** Network.
*   **Engineering Feat**: Written in **Pure NumPy**.
    *   **Forward Pass**: Custom `sigmoid` and `tanh` activations, gate calculations (Input, Forget, Output, Cell).
    *   **Backward Pass**: Manual implementation of gradients for weights and biases.
*   **Architecture**: Embedding Layer (128-dim) $\to$ LSTM Layer (256-dim) $\to$ Dense Output Layer.
*   **Limitation**: Training on CPU with Python loops was slow, limiting us to 2 epochs, but successfully demonstrated the architecture works.

### Phase 6: Evaluation & Interface
*   **Metrics**: Perplexity, Top-1 Accuracy, Top-5 Accuracy, Mean Reciprocal Rank (MRR).
*   **Demo**: Created an interactive CLI tool (`interactive_demo.py`) for real-time testing.

---

## 4. Key Implementation References

1.  **Chandu et al. (2018)**: *Language Informed Modeling of Code-Switched Text*. Used as inspiration for the Hybrid model.
2.  **Jurafsky & Martin**: *Speech and Language Processing*. Reference for HMM and N-gram smoothing algorithms.
3.  **Colah's Blog**: *Understanding LSTM Networks*. Visual guide used for designing the NumPy LSTM cells.

---

## 5. How to Replicate

1.  **Clone the Repo**:
    ```bash
    git clone https://github.com/chidwipaK/Tenglish-CodeMix-Predictor
    ```
2.  **Run Interactive Demo**:
    ```bash
    python interactive_demo.py
    ```
3.  **Train Models**:
    ```bash
    python train_all_models.py
    ```

---

*This project represents a comprehensive exploration of NLP fundamentals, moving from statistical probability to deep learning, all grounded in the challenging domain of code-mixed Indian languages.*
