import random
from collections import defaultdict, Counter
import numpy as np


def build_ngram_model(corpus, n):
    model = defaultdict(Counter)
    
    for i in range(len(corpus) - n + 1):
        ngram = tuple(corpus[i:i+n])
        context = ngram[:-1]
        next_token = ngram[-1]
        model[context][next_token] += 1
    
    return model


def get_next_token_probabilities(context, corpus, n, alpha=0.4):
    for context_len in range(min(len(context), n-1), -1, -1):
        if context_len == 0:
            return Counter(corpus)
        
        ngram_order = context_len + 1
        model = build_ngram_model(corpus, ngram_order)
        
        current_context = context[-context_len:]
        
        if current_context in model:
            counts = model[current_context]
            backoff_penalty = alpha ** (len(context) - context_len)
            
            result = Counter()
            for token, count in counts.items():
                result[token] = count * backoff_penalty
            
            return result
    
    # Shouldn't reach here but just in case
    return Counter(corpus)


def choose_next_token(probabilities, randomize=False):
    """Pick next token - deterministic or random."""
    if not probabilities:
        return None
    
    if randomize:
        tokens = list(probabilities.keys())
        weights = list(probabilities.values())
        total = sum(weights)
        weights = [w/total for w in weights]
        return np.random.choice(tokens, p=weights)
    else:
        # Deterministic
        max_prob = max(probabilities.values())
        candidates = [token for token, prob in probabilities.items() if prob == max_prob]
        return min(candidates)


def finish_sentence(sentence, n, corpus, randomize=False, alpha=0.4):
    result = list(sentence)
    sentence_enders = {'.', '?', '!'}
    
    while len(result) < 10:
        # Get context (last n-1 tokens)
        max_context_len = n - 1
        context = tuple(result[-max_context_len:]) if len(result) >= max_context_len else tuple(result)
        
        # Get probabilities using stupid backoff
        probabilities = get_next_token_probabilities(context, corpus, n, alpha)
        
        next_token = choose_next_token(probabilities, randomize)
        
        if next_token is None:
            break
        
        result.append(next_token)

        if next_token in sentence_enders:
            break
    
    return result


def run_example_applications():
    import nltk
    
    try:
        corpus = tuple(nltk.word_tokenize(nltk.corpus.gutenberg.raw('austen-sense.txt').lower()))
    except LookupError:
        print("Need to download NLTK data first")
        return
    
    print("Markov Text Generator Examples")
    print("=" * 40)
    print()
    
    print("Assignment test case:")
    sentence = ['she', 'was', 'not']
    n = 3
    result = finish_sentence(sentence, n, corpus, randomize=False)
    print(f"Result: {result}")
    print()
    
    # Different seeds and n values
    print("Various examples:")
    test_cases = [
        (['it', 'was'], 2),
        (['he', 'said'], 2), 
        (['they', 'were', 'very'], 3),
        (['in', 'the', 'morning'], 4)
    ]
    
    for seed, n in test_cases:
        print(f"Seed: {seed}, n={n}")
        
        # Deterministic
        det_result = finish_sentence(seed, n, corpus, randomize=False)
        print(f"  Det: {' '.join(det_result)}")
        
        # Random 
        rand_result = finish_sentence(seed, n, corpus, randomize=True)
        print(f"  Rand: {' '.join(rand_result)}")
        print()

if __name__ == "__main__":
    run_example_applications()