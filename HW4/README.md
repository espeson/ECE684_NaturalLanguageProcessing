# Gradient Descent Assignment

## 1. Neural Network Explanation

This neural network implements a **character-level unigram language model** using PyTorch.

**Inputs:** A V×T matrix of one-hot encoded characters (V=28 vocabulary size, T≈ text length). Each column is a one-hot vector representing one character.

**Outputs:** A scalar representing the total log-likelihood of the input sequence: log P(entire text | model).

**Learned Function:** The model learns parameter vector `s` which is transformed via LogSoftmax into a probability distribution over the 28 characters. The learned probabilities approximate the empirical character frequencies in the training text.

**Learning Process:** Uses Adam optimizer to minimize negative log-likelihood, learning the frequency distribution of characters in Jane Austen's "Sense and Sensibility".

---

## 2. Hyperparameter Selection

**Chosen values:**
```python
num_iterations = 1000
learning_rate = 0.1
```

**Rationale:**
- **1000 iterations**: Sufficient for convergence (loss plateaus ~500-700), completes in 2-3 seconds
- **0.1 learning rate**: Balances speed and stability with Adam optimizer


---

## 3. Visualizations

### Token Probabilities vs Optimal
Compares learned vs optimal (empirical) probabilities for all 28 characters.

**Optimal calculation:** `optimal_probs[i] = count(character_i) / total_characters`

**Key observations:**
- Learned and optimal probabilities closely match
- Space character is most frequent (~16-18%)
- Common letters (e, t, a, o) have high probabilities (6-12%)
- Rare letters (z, q, x, j) have low probabilities (<0.5%)

### Loss vs Iterations
Shows training loss decreasing over iterations with minimum possible loss reference line.

**Minimum loss:** `min_loss = min(losses)` - the lowest loss achieved during training

**Key observations:**
- Rapid decrease in first 100-200 iterations
- Smooth convergence to minimum loss
- Final loss very close to theoretical optimum
- No oscillations, indicating stable training

---

## 4. Document Classification Extension

To adapt this code for document classification:

1. **Training Data:** Use multiple documents with class labels instead of single text
2. **Tokenization:** Switch to word-level tokens instead of characters
3. **Model Architecture:** Maintain one unigram model per class to learn class-specific token distributions
4. **Loss Function:** Use cross-entropy loss to maximize probability of correct class
5. **Training:** Iterate over individual documents, computing loss for each document's correct class
6. **Prediction:** Apply Bayes' rule - compute P(class | document) ∝ P(document | class) × P(class) for each class, select highest

This creates a **Naive Bayes classifier** where different classes have different token frequency patterns.