# Spelling Corrector Analysis

## Modeling Assumptions

1. **Character-Level Independence**: Edit operations are independent of broader context, only considering immediate character neighbors

2. **Static Language Model**: Uses fixed unigram word frequencies, assuming language usage patterns remain constant

3. **Zero Probability for Unseen Events**: Unseen edit operations have probability 0 (no smoothing applied)

4. **Complete Vocabulary**: Target words must exist in the frequency dictionary to be considered as corrections

## Scenarios Where the Corrector Works Well

### Single-Character Substitution
- **Example**: `'helo' → 'help'`
- **Why it works**: One substitution operation, high frequency target word

### Single-Character Deletion  
- **Example**: `'pytho' → 'python'`
- **Why it works**: One insertion operation to recover common word

### Single-Character Insertion
- **Example**: `'googlee' → 'google'` 
- **Why it works**: One deletion operation, very high frequency target word

## Scenarios Where It Could Do Better

### Transposition Errors
- **Example**: `'chekc' → 'checks'` (should be `'check'`)
- **Problem**: Adjacent character swaps require two operations (delete + insert), exceeds single-edit assumption

### Multiple Edit Errors
- **Example**: `'acomodate' → 'accomodate'` (should be `'accommodate'`)  
- **Problem**: Requires multiple insertions, but algorithm only considers single edits

### Context-Sensitive Errors
- **Example**: `'there' → 'thee'` (should be `'their'` in context "their house")
- **Problem**: Both words are valid, needs contextual understanding which unigram model cannot provide

### Out-of-Vocabulary Words
- **Example**: `'gpoll' → 'groll'` (should be `'apollo'` if in vocabulary)
- **Problem**: Target word may not exist in training vocabulary, gets zero probability

## Why Poor Decisions Occur

1. **Limited Search Space**: Only considers candidates within one edit distance
2. **No Context Awareness**: Cannot use surrounding words to disambiguate
3. **Vocabulary Gaps**: Missing or low-frequency words get penalized
4. **Zero Probability Problem**: Unseen patterns eliminated completely

## Improvement Suggestions

1. **Add Transposition Operations**: Include adjacent character swaps in candidate generation

2. **Implement Smoothing**: Use add-alpha smoothing for unseen edit operations to avoid zero probabilities

3. **Context Integration**: Upgrade to bigram/trigram language models to consider word sequences

4. **Extended Search**: Allow limited multi-edit corrections for short words using beam search

5. **Dynamic Vocabulary**: Enable online learning from user corrections and domain-specific dictionaries