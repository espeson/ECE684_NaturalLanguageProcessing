# Markov Text Generation Report

## Implementation

### Function Signature
```python
finish_sentence(sentence, n, corpus, randomize=False, alpha=0.4)
```

### Algorithm
Uses n-gram models with stupid backoff (α=0.4):
1. Try full context (n-1 tokens)
2. If no match, reduce context and apply penalty
3. Final fallback: use corpus word frequencies

### Token Selection
- **Deterministic**: Highest probability, alphabetical tiebreaker
- **Stochastic**: Random sampling by probability distribution

## Test Results


### Example Applications

**Different seeds and n-values:**

1. `['it', 'was']`, n=2
   - Det: "it was not be a very well , and the"
   - Rand: "it was then , elinor was very good news with"

2. `['he', 'said']`, n=2
   - Det: "he said elinor , and the same time , and"
   - Rand: "he said t'other day or other gentlemen arrived to harley"

3. `['they', 'were', 'very']`, n=3
   - Det: "they were very , very well , and the two"
   - Rand: "they were very dark , there was any news ."

4. `['in', 'the', 'morning']`, n=4
   - Det: "in the morning , dined with them ."
   - Rand: "in the morning , early enough to interrupt the lovers"
