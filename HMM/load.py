import numpy as np
import os



# Read training corpus
def get_training_corpus(corpus_url):
    with open(corpus_url, 'r') as f:
        training_corpus = f.readlines()
        return training_corpus


def get_index_vocab(vocab_txt, verbose = True):
    """
    Get vocab dictionary from vocab txt
    Input: vocab txt url
    Output: vocab{word0: 0, word1: 1, ...}
    """
    with open(vocab_txt, 'r') as f:
        voc_l = f.read().split('\n')    
    # Remove duplicate
    voc_l = set(voc_l)
    vocab = {}
    for i, word in enumerate(sorted(voc_l)): 
        vocab[word] = i
    if verbose:       
        print("Vocabulary dictionary which length of {} is created!".format(len(vocab)))
    return vocab
if __name__ == "__main__":
    vocab = get_index_vocab("./data/COCA60000.txt")
    print(len(set(vocab)))


