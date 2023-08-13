import numpy as np 
import string 
from load_vocab import get_index_vocab
from utils import *
# Punctuation characters
punct = set(string.punctuation)



def preprocess(vocab, test_words):

    origin = []
    processed = []
    print(len(vocab))
    with open(test_words, "r") as data_file:
        for _, word in enumerate(data_file):
            i += 1
            # End of sentence
            if not word.split():
                origin.append(word.strip())
                word = "--n--"
                processed.append(word)
                continue

            # Handle unknown words
            elif word.strip() not in vocab:
                origin.append(word.strip())
                word = assign_unk(word)
                processed.append(word)
                continue

            else:
                origin.append(word.strip())
                processed.append(word.strip())
    return origin, processed

if __name__ == "__main__":
    org, prep = preprocess(get_index_vocab("./data/vocab.txt"), "./data/test.words")
    print(org[:50])
    print(prep[:50])
