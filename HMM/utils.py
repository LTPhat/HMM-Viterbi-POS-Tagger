import numpy as np
import string
from load import *
punct = set(string.punctuation)

noun_suffix = ["action", "al", "age", "ance", "cy", "dom", "ee", "ence", "er", "hood", "ion", "ism", "ist", "ity", "ling", "ment", "ness", "or", "ry", "scape", "ship", "ty"]
verb_suffix = ["ate", "ify", "ise", "ize", "en"]
adj_suffix = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive", "less", "ly", "ous"]
adv_suffix = ["ward", "wards", "wise"]


def assign_unk(token):
    """
    Assign unknown word (not in vocab) tokens
    """
    # Digits
    if any(char.isdigit() for char in token):
        return "--unk_digit--"

    # Punctuation
    elif any(char in punct for char in token):
        return "--unk_punct--"

    # Upper-case
    elif any(char.isupper() for char in token):
        return "--unk_upper--"

    # Nouns
    elif any(token.endswith(suffix) for suffix in noun_suffix):
        return "--unk_noun--"

    # Verbs
    elif any(token.endswith(suffix) for suffix in verb_suffix):
        return "--unk_verb--"

    # Adjectives
    elif any(token.endswith(suffix) for suffix in adj_suffix):
        return "--unk_adj--"

    # Adverbs
    elif any(token.endswith(suffix) for suffix in adv_suffix):
        return "--unk_adv--"

    return "--unk--"

def get_word_tag(line, vocab):
    """
    Get word and tag in a line of train/test corpus
    Input: Word \ t \Tag \ n
    Output: Word, Tag
    """ 
    if not line.split():
        word = "--n--"
        tag = "--s--"
        return word, tag
    else:
        word, tag = line.split()
        if word not in vocab: 
            # Handle unknown words
            word = assign_unk(word)
        return word, tag
    return None 

def preprocess(vocab, corpus):
    """
    Preprocess test corpus
    """
    origin = []
    processed = []
    with open(corpus, "r") as data_file:
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
    vocab = "./data/hmm_vocab.txt"
    test_corpus = "./data/WSJ_24.pos"
    orgi, processed = preprocess(vocab=get_index_vocab(vocab), corpus = test_corpus)
    print(len(processed))
    print(processed[:10])
    print(orgi[:10])
    print(list(get_index_vocab(vocab_txt=vocab).items())[:10])