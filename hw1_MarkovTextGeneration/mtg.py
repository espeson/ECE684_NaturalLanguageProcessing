import random
from collections import defaultdict, Counter

def finish_sentence(sentence, n, corpus, randomize=False):
    """
    Finish a sentence using a n-gram model with stupid backoff.

    Args:
        sentence (tuple): The input sentence (tuple of tokens).
        n (int): The length of the n-grams.
        corpus (tuple): The source corpus (tuple of tokens).
        randomize (bool): Whether to randomly select the next word.

    Returns:
        result (list): The completed sentence (list of tokens).
    """

    def get_ngrams(tokens, n):
        """
        Generate n-grams from a list of tokens.

        Args:
            tokens (list): A list of tokens.
            n (int): The size of the n-grams.
        
        Returns:
            ngrams (list of tuples): A list of n-grams.
        """
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams.append(ngram)
        return ngrams

    def build_model(corpus, n):
        """
        Build an n-gram model from a corpus.

        Args:
            corpus (list): A list of tokens.
            n (int): The size of the n-grams (n > 1, no unigrams).

        Returns:
            model (defaultdict): The n-gram language model - a dictionary of n-gram histories.
        """
        if n < 2:
            raise ValueError("n must be greater than 1, i.e., no unigrams.")
        
        model = defaultdict(Counter)
        for ngram in get_ngrams(corpus, n):
            history, token = tuple(ngram[:-1]), ngram[-1]
            model[history][token] += 1
        return model

    def stupid_backoff(history, alpha=0.4):
        """
        Implement the stupid backoff algorithm to generate the next word.
        
        Args:
            history (tuple): The previous n-1 words in the sequence.
            alpha (float): The backoff factor, default is 0.4.
        
        Returns:
            A dictionary of possible next words and their adjusted counts.
        
        The function works as follows:
        1. Start with the full history.
        2. If the full history is found in the model, return the corresponding word distribution.
        3. If not, back off to a shorter history and multiply counts by alpha.
        4. Repeat steps 2-3, progressively shortening the history and applying alpha each time.
        5. If no match is found even for bigrams, return an unigram model distribution. 
        """
        for i in range(len(history), 0, -1):
            n = i + 1
            model = build_model(corpus, n)
            if history[-i:] in model:
                return {word: count * (alpha ** (len(history) - i))
                        for word, count in model[history[-i:]].items()}
        return Counter(corpus)

    # Initialize the result with the input sentence.
    result = list(sentence)

    while len(result) < 10 and result[-1] not in '.?!':
        history = tuple(result[-(n-1):]) if len(result) >= n-1 else tuple(result)
        distribution = stupid_backoff(history)

        # Stochastic Mode: Choose the next word randomly based on the distribution.
        if randomize:
            total = sum(distribution.values())
            rand = random.random() * total
            for word, count in distribution.items():
                rand -= count
                if rand <= 0:
                    next_word = word
                    break
        # Deterministic Mode: Choose the most likely next word, breaking ties alphabetically.
        else:
            max_count = max(distribution.values())
            next_word = min([word for word, count in distribution.items() if count == max_count])

        result.append(next_word)

    return result

# Example applications of the finish_sentence function in both deterministic and stochastic modes.
if __name__ == "__main__":
    import nltk

    corpus = tuple(
        nltk.word_tokenize(nltk.corpus.gutenberg.raw("austen-sense.txt").lower())
    )

    # Seed words: 'she was not'
    # Test case 1: Deterministic
    sentence = ['she', 'was', 'not']
    n = 3
    randomize = False
    result = finish_sentence(sentence, n, corpus, randomize)
    print(f"Deterministic (n={n}): {' '.join(result)}")

    # Test case 2: Stochastic
    sentence = ['she', 'was', 'not']
    n = 3
    randomize = True
    for _ in range(3):
        result = finish_sentence(sentence, n, corpus, randomize)
        print(f"Stochastic (n={n}): {' '.join(result)}")

    # Test case 3: Different n-gram size (n = 4) and deterministic mode
    sentence = ['she', 'was', 'not']
    n = 4
    randomize = False
    result = finish_sentence(sentence, n, corpus, randomize)
    print(f"Deterministic (n={n}): {' '.join(result)}")

    # Test case 4: Different n-gram size (n = 4) and stochastic mode
    sentence = ['she', 'was', 'not']
    n = 4
    randomize = True
    for _ in range(3):
        result = finish_sentence(sentence, n, corpus, randomize)
        print(f"Stochastic (n={n}): {' '.join(result)}")

    # Seed words: 'could not even'
    # Test case 5: Deterministic
    sentence = ['could', 'not', 'even']
    n = 3
    randomize = False
    result = finish_sentence(sentence, n, corpus, randomize)
    print(f"Deterministic (n={n}): {' '.join(result)}")
    
    # Test case 6: Stochastic
    sentence = ['could', 'not', 'even']
    n = 3
    randomize = True
    for _ in range(3):
        result = finish_sentence(sentence, n, corpus, randomize)
        print(f"Stochastic (n={n}): {' '.join(result)}")

    # Test case 7: Different n-gram size (n = 4) and deterministic mode
    sentence = ['could', 'not', 'even']
    n = 4
    randomize = False
    result = finish_sentence(sentence, n, corpus, randomize)
    print(f"Deterministic (n={n}): {' '.join(result)}")

    # Test case 8: Different n-gram size (n = 4) and stochastic mode
    sentence = ['could', 'not', 'even']
    n = 4
    randomize = True
    for _ in range(3):
        result = finish_sentence(sentence, n, corpus, randomize)
        print(f"Stochastic (n={n}): {' '.join(result)}")