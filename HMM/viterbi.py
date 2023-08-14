import numpy as np
from hmm import HMM
from load import *
from utils import *
import math

class Viterbi(object):
    def __init__(self, vocab_txt, tag_counts, transition_matrix, emission_matrix, test_corpus):
        """
        vocab_txt: Vocabulary txt file
        tag_counts: Dict of tag_counts (like HMM class)
        transition matrix: transition matrix of HMM class
        emission matrix: emission matrix of HMM class
        test_corpus: test corpus (include test words and according true labels)
        """
        self.vocab = get_index_vocab(vocab_txt=vocab_txt)
        self.tag_counts = tag_counts
        self.transition_matrix = transition_matrix
        self.emission_matrix = emission_matrix
        self.test_corpus = test_corpus

        # Inner attributes
        """
        states: List of all possible tags
        test_word: file of test words (.words) after remove true labels
        """
        self.states = list(sorted(self.tag_counts.keys()))
        self.test_words = preprocess(vocab=self.vocab, corpus=self.test_corpus)[1]
        # best_probs has shape of (num_tags, len(test_corpus)) of float:  
        # best_probs[i, j] gives the probs at which word[j] is assigned with tag i 
        self.best_probs = np.zeros((len(self.tag_counts), len(self.test_words)))           
        # best_path has (num_tags, len(test_corpus)) of integer store index of assign tag in vocab
        self.best_paths = np.zeros((len(self.tag_counts), len(self.test_words)), dtype = int)
    

    def _initialize(self):
        """
        Calculate column 0 (prob of start token - first word)
        best_probs[i, 0] = log(A[start_token_idx, i]) + log(B[i, vocab[test_word[0]]]) 
        """
        start_token = "--s--"
        start_token_idx = self.states.index(start_token)
        num_tags = len(self.tag_counts)
        for i in range(num_tags):
            self.best_probs[i, 0] = math.log(self.transition_matrix[start_token_idx, i]) + math.log(self.emission_matrix[i, self.vocab[self.test_words[0]]])
        return 
    

    def _forward(self, verbose = True):
        """
        Complete best_probs, best_paths matrix
        """
        num_tags = len(self.tag_counts)
        # Traverse each word in test_words from 1, because index 0 is assigned for start_token 
        for i in range(1, len(self.test_words)):
            # Print number of words processed, every 5000 words
            if i % 5000 == 0 and verbose:
                print("Words processed: {:>8}".format(i))
            # Traverse each num_tag at column i
            for j in range(num_tags):
                # # Normal
                best_prob = -float("inf")
                best_path = 0       
                # Traverse each num_tag at column i - 1
                # for k in range(num_tags):
                #     prob = self.best_probs[k, i - 1] + math.log(self.transition_matrix[k, j]) + math.log(self.emission_matrix[j, self.vocab[self.test_words[i]]])
                #     # Update best_prob
                #     if prob > best_prob:
                #         best_prob = prob
                #         best_path = k
                # Vectorization (speed up x3)
                prob_vector_i = self.best_probs[:, i - 1] + np.log(self.transition_matrix[:, j]) +  np.log(self.emission_matrix[j, self.vocab[self.test_words[i]]]) 
                # print(prob_vector_i)
                best_prob = np.max(prob_vector_i)
                best_path = np.argmax(prob_vector_i)
            
                self.best_probs[j, i] = best_prob
                self.best_paths[j, i] = best_path
            if i == 4:
                break

        return self.best_probs, self.best_paths
    
if __name__ == "__main__":
    hmm = HMM(training_corpus=get_training_corpus("./data/WSJ_02-21.pos"), vocab_txt="./data/hmm_vocab.txt")
    hmm._create_counts()
    hmm._create_transition_matrix()
    hmm._create_emission_matrix()
    viterbi = Viterbi(vocab_txt="./data/hmm_vocab.txt", transition_matrix=hmm.transition_matrix, 
                      emission_matrix=hmm.emission_matrix, test_corpus="./data/WSJ_24.pos", tag_counts=hmm.tag_counts)
    viterbi._initialize()
    viterbi._forward()
    print(viterbi.best_probs[0:5, 0:5])
    print(viterbi.best_probs[30:35])
    print(f"best_probs[0,1]: {viterbi.best_probs[0,1]:.4f}") 
    print(f"best_probs[0,4]: {viterbi.best_probs[0,4]:.4f}") 
    print(viterbi.best_paths[0:5, 0:5])

    