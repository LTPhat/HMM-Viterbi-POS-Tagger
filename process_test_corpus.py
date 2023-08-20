import numpy as np 
import string 
from utils import *


# Punctuation characters
punct = set(string.punctuation)


def load_test_corpus(corpur_url):
    """
    Split test corpus
    Input: test_corpus url
    Output:
    - test_words: List of all words in test_corpus
    - y: List of tags according respective with each word in test_words
    """
    test_words = []
    y = []
    with open(corpur_url, 'r') as f:
        load = f.readlines()
    for item in load:
        word_tag = item.split()
        if len(word_tag) != 2:
            word = "--n--"
            tag = "--s--"
        else:
            word = word_tag[0]
            tag = word_tag[1]
        test_words.append(word)
        y.append(tag)
    return test_words, y


def preprocess_list(vocab, test_words_list):
    """
    Preprocess out of vocab with (use in case there are no test_corpus .words file)
    Input:
    - Vocab: Dict {word: index}
    - test_word_list: test_words list extracted from calling 'load_test_corpus(corpus_url)' function 
    """
    origin = []
    processed = []
    for word in test_words_list:
        if not word:
            origin.append(word.strip())
            word = "--n--"
            processed.append(word)
            continue
        elif word.strip() not in vocab:
                origin.append(word.strip())
                word = assign_unk(word)
                processed.append(word)
                continue
        else:
            origin.append(word.strip())
            processed.append(word.strip())
    return origin, processed


def preprocess_words(vocab, test_words_file):
    """
    Preprocess out of vocab with .words file
    Input:
    - Vocab: Dict {word: index}
    - test_word_file: test_corpus .words file
    """
    origin = []
    processed = []
    print(len(vocab))
    with open(test_words_file, "r") as data_file:
        for _, word in enumerate(data_file):
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
    words, label = load_test_corpus("./data/WSJ_24.pos")
    print("First 50 words in test corpus: ", words[:50])
    print("First 50 tags in test corpus: ", label[:50])
    vocab_txt="./data/hmm_vocab.txt"
    vocab = get_index_vocab(vocab_txt=vocab_txt)
    _, test_words = preprocess_list(vocab=vocab, test_words_list=words) 
    print("First 50 words in test corpus after processing: ", test_words[:30])

