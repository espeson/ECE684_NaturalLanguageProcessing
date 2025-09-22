import numpy as np
import nltk
from collections import Counter
from viterbi import viterbi


# nltk.download('brown')
# nltk.download('universal_tagset')

def build_hmm_components(tagged_sentences):
    
    # extract tags and words
    all_states = set()
    all_observations = set()
    
    for sentence in tagged_sentences:
        for word, tag in sentence:
            all_states.add(tag)
            all_observations.add(word.lower())

    # add unknown token for unseen words
    all_observations.add('<UNK>')
    
    states = sorted(list(all_states))
    observations = sorted(list(all_observations))
    
    state_to_idx = {state: idx for idx, state in enumerate(states)}
    idx_to_state = {idx: state for idx, state in enumerate(states)}
    obs_to_idx = {obs: idx for idx, obs in enumerate(observations)}
    idx_to_obs = {idx: obs for idx, obs in enumerate(observations)}
    
    unk_idx = obs_to_idx['<UNK>']
    
    # add-1 smoothing
    num_states = len(states)
    num_observations = len(observations)

    transition_counts = np.ones((num_states, num_states))
    observation_counts = np.ones((num_states, num_observations))
    initial_counts = np.ones(num_states)

    for sentence in tagged_sentences:
        if not sentence:
            continue
            
        # initial state
        first_word, first_tag = sentence[0]
        initial_counts[state_to_idx[first_tag]] += 1
        
        # first word observation
        prev_state_idx = state_to_idx[first_tag]
        word_idx = obs_to_idx.get(first_word.lower(), unk_idx)
        observation_counts[prev_state_idx, word_idx] += 1
        
        # transitions and observations for rest of sentence
        for i in range(1, len(sentence)):
            word, tag = sentence[i]
            curr_state_idx = state_to_idx[tag]
            word_idx = obs_to_idx.get(word.lower(), unk_idx)
            
            # transition
            transition_counts[prev_state_idx, curr_state_idx] += 1
            
            # observation
            observation_counts[curr_state_idx, word_idx] += 1
            
            prev_state_idx = curr_state_idx

    transition_matrix = transition_counts / transition_counts.sum(axis=1, keepdims=True)
    observation_matrix = observation_counts / observation_counts.sum(axis=1, keepdims=True)
    initial_state_dist = initial_counts / initial_counts.sum()
    
    return (transition_matrix, observation_matrix, initial_state_dist, 
            states, observations, state_to_idx, idx_to_state, obs_to_idx, idx_to_obs, unk_idx)

def predict_pos_tags(words, transition_matrix, observation_matrix, initial_state_dist,
                    obs_to_idx, idx_to_state, unk_idx):

    # convert words to observation
    obs_indices = []
    for word in words:
        word_lower = word.lower()
        if word_lower in obs_to_idx:
            obs_indices.append(obs_to_idx[word_lower])
        else:
            obs_indices.append(unk_idx)

    state_sequence, probability = viterbi(
        obs=obs_indices,
        pi=initial_state_dist,
        A=transition_matrix,
        B=observation_matrix
    )
    
    # convert state indices back to POS tag names
    tag_sequence = [idx_to_state[state_idx] for state_idx in state_sequence]
    
    return tag_sequence, probability

def main():
    training_sentences = nltk.corpus.brown.tagged_sents(tagset='universal')[:10000]
    test_sentences = nltk.corpus.brown.tagged_sents(tagset='universal')[10150:10153]

    (transition_matrix, observation_matrix, initial_state_dist,
     states, observations, state_to_idx, idx_to_state, obs_to_idx, idx_to_obs, unk_idx) = build_hmm_components(training_sentences)

    
    for i, sentence in enumerate(test_sentences):
        sentence_num = 10150 + i
        words = [word for word, tag in sentence]
        true_tags = [tag for word, tag in sentence]
        
        print(f"\nSentence {sentence_num}:")
        print(f"Words: {' '.join(words)}")
        

        predicted_tags, probability = predict_pos_tags(
            words, transition_matrix, observation_matrix, initial_state_dist,
            obs_to_idx, idx_to_state, unk_idx
        )
        
        print(f"True tags:      {' '.join(true_tags)}")
        print(f"Predicted tags: {' '.join(predicted_tags)}")

if __name__ == "__main__":
    main()