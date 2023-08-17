#!/bin/bash


# You can adjust params then run bash script 

python main.py --training_corpus './data/WSJ_02-21.pos' --vocab_txt './data/hmm_vocab.txt' --test_corpus './data/WSJ_24.pos' --alpha 0.001

python predict_new.py --training_corpus "./data/WSJ_02-21.pos" --vocab_txt "./data/hmm_vocab.txt" --alpha 0.001 --stn "I am a student at school."

