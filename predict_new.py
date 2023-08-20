import numpy as np
import os
import pandas as pd

from load import *
from utils import *
from hmm import HMM
from viterbi import Viterbi
from process_test_corpus import *
import argparse
from main import vocab_txt_list



def process_new_stn(vocab, new_sentence):
    # Remove white space at head and tail
    new_sentence = new_sentence.strip()
    if new_sentence[-1] == ".":
        # Tokenize
        test_words = new_sentence[:-1].split(" ")
        test_words.append(new_sentence[-1])
    else: 
        test_words = new_sentence.split(" ")
    # Out of vocab preprocess
    org_test_words, test_words = preprocess_list(vocab=vocab, test_words_list=test_words)
    return org_test_words, test_words


def predict_new(vocab, tag_counts, transition_matrix, emission_matrix, test_word, y):
    viterbi = Viterbi(vocab=vocab, tag_counts= tag_counts, transition_matrix= transition_matrix,
                       emission_matrix=emission_matrix, test_words=test_word, y=y)
    viterbi._initialize()
    viterbi._forward()
    pred = viterbi._backward()
    return pred


def check_valid_vocab_txt(args):
    if args.vocab_txt not in vocab_txt_list:
        print("Wrong vocabulary txt file to create vocab, expected one of {}".format(vocab_txt_list))
        return False
    return True


def init_argparse():
    parser = argparse.ArgumentParser(
        prog= "Predict new sentence",
        usage="%(prog)s --training_corpus '.data/WSJ_02-21.pos' --vocab_txt './data/hmm_vocab.txt' --alpha 0.001 --stn 'Your sentence' ",
        description=
                    "--vocab_txt: Vocab txt file to refer. There are 5k vocab (./data/WordList_TOEFL.txt); 20k vocab (./data/hmm_vocab.txt); 60k vocab (./data/COCA60000_vocab.txt); 500k vocab (./data/vocab.txt). Choose one of these. \n"
                    "--alpha: Laplacian smoothing coefficient of HMM model. Alpha = 0.001 (default)"
                    "--stn: Your new sentence to predict tags"
    )   
    parser.add_argument("--training_corpus", required=True, 
        help='Training corpus .pos file, please Enter this: ./data/WSJ_02-21.pos')
    parser.add_argument(
        "--vocab_txt", required=True,
        help='Vocab txt file in data folder. Choose one of these (./data/WordList_TOEFL.txt, ./data/hmm_vocab.txt, ./data/COCA60000_vocab.txt, ./data/vocab.txt)'
    )
    parser.add_argument(
        "--alpha",required=False ,default = 0.001,
        help='Laplacian Smoothing Coefficient of HMM model'
    )
    parser.add_argument(
        "--stn", required=True,
        help='Your sentence to predict tags'
    )
    return parser


def main_predict():
    print("-------- YOUR SENTENCE TAGGING IS PROCESSING------------")
    parser = init_argparse()
    args   = parser.parse_args()
    if not check_valid_vocab_txt(args=args):
        print("-------Received parser vocab failed-----------")
        return 

    print("-------Received parser inputs completed-----------")

    # Define variables
    training_corpus = args.training_corpus
    vocab_txt = args.vocab_txt
    alpha = float(args.alpha)
    new_stn = args.stn
    # Preprocess
    training_corpus = get_training_corpus(training_corpus)
    vocab = get_index_vocab(vocab_txt= vocab_txt, verbose=False)
    hmm = HMM(training_corpus=training_corpus, vocab=vocab, alpha=alpha)
    hmm._create_counts()
    hmm._create_transition_matrix()
    hmm._create_emission_matrix()
    org, test_words = process_new_stn(vocab=vocab, new_sentence=new_stn)
    # Predict new sentence
    pred = predict_new(vocab=vocab, tag_counts=hmm.tag_counts, transition_matrix=hmm.transition_matrix,
                       emission_matrix=hmm.emission_matrix, test_word=test_words,y = None)

    print("Sentence: {}".format(org))
    print("POS tags: {}".format(pred))
    return pred


if __name__ == "__main__":
    main_predict()
    



