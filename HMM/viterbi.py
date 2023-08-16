import numpy as np
from hmm import HMM
from load import *
from utils import *
import math
from process_test_corpus import *
import os
import pickle


class Viterbi(object):
    def __init__(self, vocab_txt, tag_counts, transition_matrix, emission_matrix, test_corpus):
        """
        vocab_txt: Vocabulary txt file
        tag_counts: Dict of tag_counts (like HMM class)
        transition matrix: transition matrix of HMM class
        emission matrix: emission matrix of HMM class
        test_corpus: test corpus (include test words and according true labels) (.pos)
        """
        self.vocab_txt = vocab_txt
        self.tag_counts = tag_counts
        self.transition_matrix = transition_matrix
        self.emission_matrix = emission_matrix
        self.test_corpus = test_corpus

        # Inner attributes (processed by calling Set_up method)
        """
        states: List of all possible tags
        test_word: file of test words (.words) after remove true labels
        """
        self.vocab = get_index_vocab(vocab_txt=self.vocab_txt)
        self.states = list(sorted(self.tag_counts.keys()))
        self.test_words, self.y = load_test_corpus(self.test_corpus)
        self.best_probs = np.zeros((len(self.tag_counts), len(self.test_words)))     
        self.best_paths = np.zeros((len(self.tag_counts), len(self.test_words)), dtype = int)
        self.pred = [None] * len(self.test_words)


    def Set_up(self):
        """
        Preprocess attributes assign to class
        """
        # Create vocab with index
        self.vocab = get_index_vocab(vocab_txt=self.vocab_txt)

        # Extract test_words and tags respectively
        self.test_words, self.y = load_test_corpus(self.test_corpus)
        _, self.test_words = preprocess_list(vocab=self.vocab, test_words_list=self.test_words)

        # Nums of all possible tags
        self.states = list(sorted(self.tag_counts.keys()))

        self.best_probs = np.zeros((len(self.tag_counts), len(self.test_words)))     
        self.best_paths = np.zeros((len(self.tag_counts), len(self.test_words)), dtype = int)
        self.pred = [None] * len(self.test_words)

    def _initialize(self):
        """
        Calculate column 0 (prob of start token - first word)
        best_probs[i, 0] = log(A[start_token_idx, i]) + log(B[i, vocab[test_word[0]]]) 
        """
        start_token = "--s--"
        start_token_idx = self.states.index(start_token)
        num_tags = len(self.tag_counts)
        for i in range(num_tags):
            self.best_probs[i, 0] = np.log(self.transition_matrix[start_token_idx, i]) + np.log(self.emission_matrix[i, self.vocab[self.test_words[0]]])
        return self.best_probs, self.best_paths
    

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
                # best_prob = -float("inf")
                # best_path = 0       
                # # Traverse each num_tag at column i - 1
                # for k in range(num_tags):
                #     prob = self.best_probs[k, i - 1] + math.log(self.transition_matrix[k, j]) + math.log(self.emission_matrix[j, self.vocab[self.test_words[i]]])
                #     # Update best_prob
                #     if prob > best_prob:
                #         best_prob = prob
                #         best_path = k
                # Vectorization (speed up x5)
                prob_vector_i = self.best_probs[:, i - 1] + np.log(self.transition_matrix[:, j]) +  np.log(self.emission_matrix[j, self.vocab[self.test_words[i]]]) * np.ones((num_tags, 1))
                best_prob = np.max(prob_vector_i)
                best_path = np.argmax(prob_vector_i)
                
                self.best_probs[j, i] = best_prob
                self.best_paths[j, i] = best_path
            # if i == 4:
            #     break

        return self.best_probs, self.best_paths
    
    
    def _backward(self):
        n = len(self.test_words)
        # Array z store the decision of each colummn
        z = [None] * n
        num_tags = len(self.tag_counts)
        # Find largest probability of last column
        z[n - 1] = np.argmax(self.best_probs[:, n - 1])
        self.pred[n - 1] = self.states[z[n - 1]]

        for i in range(n - 1, -1, - 1):
            pos_tag_for_word_i = z[i]
            z[i - 1] = self.best_paths[pos_tag_for_word_i, i]
            self.pred[i - 1] = self.states[z[i - 1]]
        return self.pred
    

    def _calculate_accuracy(self):
        """
        Calculate accuracy of self.pred and self.y
        """
        assert len(self.pred) == len(self.y)
        num_correct = 0
        for pred, label in zip(self.pred, self.y):
            if (pred == label):
                num_correct += 1
        return num_correct / len(self.pred)
    
    @staticmethod
    def save_data(data, save_dir, data_name):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(save_dir + "/{}".format(data_name), data)
        return 

    
if __name__ == "__main__":
    hmm = HMM(training_corpus=get_training_corpus("./data/WSJ_02-21.pos"), vocab_txt="./data/hmm_vocab.txt", alpha=0.001)
    hmm._create_counts()
    hmm._create_transition_matrix()
    hmm._create_emission_matrix()
    viterbi = Viterbi(vocab_txt="./data/hmm_vocab.txt", transition_matrix=hmm.transition_matrix, 
                      emission_matrix=hmm.emission_matrix, test_corpus="./data/WSJ_24.pos", tag_counts=hmm.tag_counts)
    viterbi.Set_up()
    best_probs_init, best_paths_init = viterbi._initialize()

    # viterbi.save_data(best_probs_init, "./npy", "best_probs_init")
    # viterbi.save_data(best_paths_init, "./npy", "best_paths_init")

    best_probs, best_paths = viterbi._forward()
     # viterbi.save_data(best_probs, "./npy", "best_probs")
    # viterbi.save_data(best_paths, "./npy", "best_paths")

    viterbi._backward()
    print(viterbi._calculate_accuracy())

    