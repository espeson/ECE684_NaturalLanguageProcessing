import csv
from collections import defaultdict
from string import ascii_lowercase as ALPHABET

def build_error_statistics():
    del_map = defaultdict(int)
    ins_map = defaultdict(int)
    sub_map = defaultdict(int)
    bi_map = defaultdict(int)
    uni_map = defaultdict(int)

    # 加载删除错误
    with open("deletions.csv", mode="r", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # 跳过表头
        for pre, char_removed, freq in csv_reader:
            token = pre + char_removed
            del_map[token] += int(freq)

    # 加载插入错误
    with open("additions.csv", mode="r", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for pre, char_added, freq in csv_reader:
            token = pre + char_added
            ins_map[token] += int(freq)

    # 加载替换错误
    with open("substitutions.csv", mode="r", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for src_char, tgt_char, freq in csv_reader:
            token = tgt_char + src_char
            sub_map[token] += int(freq)

    # 加载二元文法
    with open("bigrams.csv", mode="r", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for pair, freq in csv_reader:
            bi_map[pair] += int(freq)

    # 加载一元文法
    with open("unigrams.csv", mode="r", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for symbol, freq in csv_reader:
            uni_map[symbol] += int(freq)

    return del_map, ins_map, sub_map, bi_map, uni_map

# def load_error_model():
#     deletion_counts = defaultdict(int)
#     insertion_counts = defaultdict(int)
#     substitution_counts = defaultdict(int)
#     bigram_counts = defaultdict(int)
#     unigram_counts = defaultdict(int)

#     with open('deletions.csv', 'r') as f:
#         reader = csv.reader(f)
#         next(reader)
#         for prefix, deleted_char, count in reader:
#             key = prefix + deleted_char
#             deletion_counts[key] += int(count)

#     with open('additions.csv', 'r') as f:
#         reader = csv.reader(f)
#         next(reader)
#         for prefix, added_char, count in reader:
#             key = prefix + added_char
#             insertion_counts[key] += int(count)

#     with open('substitutions.csv', 'r') as f:
#         reader = csv.reader(f)
#         next(reader)  # Skip header row
#         for original_char, error_char, count in reader:
#             key = error_char + original_char
#             substitution_counts[key] += int(count)

#     with open('bigrams.csv', 'r') as f:
#         reader = csv.reader(f)
#         next(reader)  # Skip header row
#         for bigram, count in reader:
#             bigram_counts[bigram] += int(count)

#     with open('unigrams.csv', 'r') as f:
#         reader = csv.reader(f)
#         next(reader)  # Skip header row
#         for char, count in reader:
#             unigram_counts[char] += int(count)

#     return deletion_counts, insertion_counts, substitution_counts, bigram_counts, unigram_counts

# def load_language_model():
#     word_counts = defaultdict(int)
#     total_word_count = 0

#     with open('count_1w.txt', 'r') as f:
#         for line in f:
#             try:
#                 word, count = line.strip().split('\t')
#                 word_counts[word] = int(count)
#                 total_word_count += int(count)
#             except ValueError:
#                 continue

#     return word_counts, total_word_count

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

# def P_w(word, word_counts, total_word_count):
#     word_count = word_counts.get(word, 0)
#     if total_word_count == 0:
#         return 0.0
#     probability = word_count / total_word_count
#     return probability

def word_prob(term, freq_dict, total_tokens):
    count = freq_dict.get(term, 0)
    if total_tokens == 0:
        return 0.0
    return count / total_tokens


# def P_edit(edit_type, edit_info, deletion_counts, insertion_counts, substitution_counts, bigram_counts, unigram_counts):
#     if edit_type == 'deletion':
#         wi_minus_1, wi = edit_info
#         del_key = wi_minus_1 + wi
#         deletion_count = deletion_counts.get(del_key, 0)
#         bigram_count = bigram_counts.get(del_key, 0)
#         if bigram_count == 0:
#             return 0.0
#         probability = deletion_count / bigram_count
#         return probability
#     elif edit_type == 'insertion':
#         wi_minus_1, xi = edit_info
#         ins_key = wi_minus_1 + xi
#         insertion_count = insertion_counts.get(ins_key, 0)
#         unigram_count = unigram_counts.get(wi_minus_1, 0)
#         if unigram_count == 0:
#             return 0.0
#         probability = insertion_count / unigram_count
#         return probability
#     elif edit_type == 'substitution':
#         xi, wi = edit_info
#         sub_key = xi + wi
#         substitution_count = substitution_counts.get(sub_key, 0)
#         unigram_count = unigram_counts.get(wi, 0)
#         if unigram_count == 0:
#             return 0.0
#         probability = substitution_count / unigram_count
#         return probability
#     else:
#         return 0.0

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


# def get_edits(word):
#     edits = []
#     letters = 'abcdefghijklmnopqrstuvwxyz'
#     splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]

#     for L, R in splits:
#         if len(R) > 0:
#             edited_word = L + R[1:]
#             wi_minus_1 = L[-1] if L else '#'
#             wi = R[0]
#             edits.append(('deletion', (wi_minus_1, wi), edited_word))

#     for L, R in splits:
#         wi_minus_1 = L[-1] if L else '#'
#         for c in letters:
#             edited_word = L + c + R
#             xi = c
#             edits.append(('insertion', (wi_minus_1, xi), edited_word))

#     for L, R in splits:
#         if len(R) > 0:
#             for c in letters:
#                 if R[0] != c:
#                     edited_word = L + c + R[1:]
#                     xi = R[0]
#                     wi = c
#                     edits.append(('substitution', (xi, wi), edited_word))
#     return edits

def enumerate_candidate_edits(token: str):
    variants = []
    alphabet = ALPHABET
    parts = [(token[:i], token[i:]) for i in range(len(token) + 1)]

    # 删除（deletion）
    for left, right in parts:
        if right:
            candidate = left + right[1:]
            prev_char = left[-1] if left else '#'
            curr_char = right[0]
            variants.append(('deletion', (prev_char, curr_char), candidate))

    # 插入（insertion）
    for left, right in parts:
        prev_char = left[-1] if left else '#'
        for ch in alphabet:
            candidate = left + ch + right
            variants.append(('insertion', (prev_char, ch), candidate))

    # 替换（substitution）
    for left, right in parts:
        if right:
            original = right[0]
            for ch in alphabet:
                if ch != original:
                    candidate = left + ch + right[1:]
                    variants.append(('substitution', (original, ch), candidate))

    return variants

# def correct(original: str) -> str:
#     deletion_counts, insertion_counts, substitution_counts, bigram_counts, unigram_counts = build_error_statistics()
#     word_counts, total_word_count = build_language_statistics()

#     candidates = enumerate_candidate_edits(original)
#     max_probability = 0
#     best_candidate = original

#     for edit_type, edit_info, candidate in candidates:
#         word_probability = word_prob(candidate, word_counts, total_word_count)

#         if word_probability == 0:
#             continue

#         edit_probability = edit_prob(edit_type, edit_info, deletion_counts, insertion_counts, substitution_counts, bigram_counts, unigram_counts)
#         if edit_probability == 0:
#             continue

#         total_probability = word_probability * edit_probability

#         if total_probability > max_probability:
#             max_probability = total_probability
#             best_candidate = candidate

#     if max_probability == 0:
#         return original

#     return best_candidate

def pick_best_correction(src: str) -> str:
    # 载入统计（与原逻辑等价）
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



misspelled_word = 'ther'
corrected_word = pick_best_correction(misspelled_word)

print(f"Original word: {misspelled_word}")
print(f"Corrected word: {corrected_word}")