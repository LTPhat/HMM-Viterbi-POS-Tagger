import numpy as np
import string

punct = set(string.punctuation)

noun_suffix = ["action", "al", "age", "ance", "cy", "dom", "ee", "ence", "er", "hood", "ion", "ism", "ist", "ity", "ling", "ment", "ness", "or", "ry", "scape", "ship", "ty"]
verb_suffix = ["ate", "ify", "ise", "ize", "en"]
adj_suffix = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive", "less", "ly", "ous"]
adv_suffix = ["ward", "wards", "wise"]


def assign_unk(token):
    """
    Assign unknown word tokens
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