import csv
from collections import defaultdict
from string import ascii_lowercase as ALPHABET

def build_error_statistics():
    del_map = defaultdict(int)
    ins_map = defaultdict(int)
    sub_map = defaultdict(int)
    bi_map = defaultdict(int)
    uni_map = defaultdict(int)

      # additions
    with open("additions.csv", mode="r", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for pre, char_added, freq in csv_reader:
            token = pre + char_added
            ins_map[token] += int(freq)

    # deletions
    with open("deletions.csv", mode="r", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        next(csv_reader) 
        for pre, char_removed, freq in csv_reader:
            token = pre + char_removed
            del_map[token] += int(freq)

    # substitutions
    with open("substitutions.csv", mode="r", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for src_char, tgt_char, freq in csv_reader:
            token = tgt_char + src_char
            sub_map[token] += int(freq)

    # bigrams
    with open("bigrams.csv", mode="r", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for pair, freq in csv_reader:
            bi_map[pair] += int(freq)

    # unigrams
    with open("unigrams.csv", mode="r", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for symbol, freq in csv_reader:
            uni_map[symbol] += int(freq)

    return del_map, ins_map, sub_map, bi_map, uni_map



def build_language_statistics():
    freq_dict = defaultdict(int)
    token_sum = 0

    with open("count_1w.txt", mode="r", encoding="utf-8") as file:
        for record in file:
            parts = record.strip().split("\t")
            if len(parts) != 2:
                continue
            word, freq = parts
            freq_dict[word] = int(freq)
            token_sum += int(freq)

    return freq_dict, token_sum

def word_prob(term, freq_dict, total_tokens):
    count = freq_dict.get(term, 0)
    if total_tokens == 0:
        return 0.0
    return count / total_tokens




def edit_prob(op_type, details, del_map, ins_map, sub_map, bi_map, uni_map):
    if op_type == "deletion":
        prev_token, curr_token = details
        key = prev_token + curr_token
        del_count = del_map.get(key, 0)
        bi_count = bi_map.get(key, 0)
        return 0.0 if bi_count == 0 else del_count / bi_count

    elif op_type == "insertion":
        prev_token, extra_char = details
        key = prev_token + extra_char
        ins_count = ins_map.get(key, 0)
        uni_count = uni_map.get(prev_token, 0)
        return 0.0 if uni_count == 0 else ins_count / uni_count

    elif op_type == "substitution":
        wrong_char, correct_char = details
        key = wrong_char + correct_char
        sub_count = sub_map.get(key, 0)
        uni_count = uni_map.get(correct_char, 0)
        return 0.0 if uni_count == 0 else sub_count / uni_count

    return 0.0



def enumerate_candidate_edits(token: str):
    variants = []
    alphabet = ALPHABET
    parts = [(token[:i], token[i:]) for i in range(len(token) + 1)]

    # deletion
    for left, right in parts:
        if right:
            candidate = left + right[1:]
            prev_char = left[-1] if left else '#'
            curr_char = right[0]
            variants.append(('deletion', (prev_char, curr_char), candidate))

    # insertion
    for left, right in parts:
        prev_char = left[-1] if left else '#'
        for ch in alphabet:
            candidate = left + ch + right
            variants.append(('insertion', (prev_char, ch), candidate))

    # substitution
    for left, right in parts:
        if right:
            original = right[0]
            for ch in alphabet:
                if ch != original:
                    candidate = left + ch + right[1:]
                    variants.append(('substitution', (original, ch), candidate))

    return variants




def correct(src: str) -> str:
    #load stats
    del_map, ins_map, sub_map, bi_map, uni_map = build_error_statistics()
    freq_dict, token_sum = build_language_statistics()

    best_score = 0.0
    winner = src

    for op_type, op_args, cand in enumerate_candidate_edits(src):
        pw = word_prob(cand, freq_dict, token_sum)
        if pw == 0:
            continue

        pe = edit_prob(op_type, op_args, del_map, ins_map, sub_map, bi_map, uni_map)
        if pe == 0:
            continue

        score = pw * pe
        if score > best_score:
            best_score = score
            winner = cand

    return winner if best_score > 0 else src



if __name__ == "__main__":
    print("Testing Spelling Corrector")
    print("=" * 50)
    
    print("\nScenarios where it works well:")
    
    # substitution
    word = "helo"
    result = correct(word)
    print(f"'{word}' -> '{result}' (substitution error)")
    
    # deletion
    word = "pytho" 
    result = correct(word)
    print(f"'{word}' -> '{result}' (deletion error)")
    
    # insertion
    word = "googlee"
    result = correct(word)
    print(f"'{word}' -> '{result}' (insertion error)")
    
    
    print("\nScenarios where it could do better:")
    
    # Transposition (swapped letters)
    word = "chekc"
    result = correct(word)
    print(f"'{word}' -> '{result}' (should be 'check' transposition error)")
    
    # Multiple edits needed
    word = "acomodate"
    result = correct(word)
    print(f"'{word}' -> '{result}' (should be 'accommodate', needs multiple edits)")
    
    # already valid word
    word = "there"
    result = correct(word)
    print(f"'{word}' -> '{result}' (context: 'their house', but 'there' is already valid)")
    
    # Low frequency word
    word = "gpoll"
    result = correct(word)
    print(f"'{word}' -> '{result}' (should be 'gpollo' if in vocabulary)")
