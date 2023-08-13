import numpy as np
import sys
import os

# Read training corpus
# with open("./data/WSJ_02-21.pos", 'r') as f:
#     training_corpus = f.readlines()
# print(training_corpus[0:5])

def get_index_vocab(vocab_txt):
    with open(vocab_txt, 'r') as f:
        voc_l = f.read().split('\n')
    # Remove empty elements
    # while("" in voc_l):
    #     voc_l.remove("")
    
    # Get index of word in vocab
    vocab = {}
    for i, word in enumerate(sorted(voc_l)): 
        vocab[word] = i       
    print("Vocabulary dictionary created!")
    return vocab
if __name__ == "__main__":
    get_index_vocab("./data/vocab.txt")



