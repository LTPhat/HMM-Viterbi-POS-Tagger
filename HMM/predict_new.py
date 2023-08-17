import numpy as np
import os
import pandas as pd

from load import *
from utils import *
from hmm import HMM
from viterbi import Viterbi
from process_test_corpus import *
import argparse



def process_new_stn(new_sentence):
    return 


def predict_new(new_sentence):
    return 


def init_argparse():
    parser = argparse.ArgumentParser(
        prog= "Predict new sentence",
        usage="%(prog)s --vocab_txt './data/hmm_vocab.txt' --alpha 0.001 --stn 'Your sentence' ",
        description=
                    "--vocab_txt: Vocab txt file to refer. There are 5k vocab (./data/WordList_TOEFL.txt); 20k vocab (./data/hmm_vocab.txt); 60k vocab (./data/COCA60000_vocab.txt); 500k vocab (./data/vocab.txt). Choose one of these. \n"
                    "--alpha: Laplacian smoothing coefficient of HMM model. Alpha = 0.001 (default)"
                    "--stn: Your new sentence to predict tags"
    )
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

if __name__ == "__main__":
    stn = "  I want to be a friend . "
    print(stn.strip())
    print("")
