# HMM-Viterbi POS Tagger

- A traditional approach to ``Part-of-speech tagging`` problem with Hidden Markov Model and Viterbi algorithm.

## About the dataset

- This POS tagger was trained on Wall Street Journal corpus (WSJ).

- The training set is section 2-21 ``WSJ_02-21.pos``, development set is section 23 ``WSJ_23.pos`` and the test set is section 24 ``WSJ_24.pos``. All sets are saved in ``data`` folder.

- Part-of-speech tags are defined in ``Penn Treebank II tag set``. [Here for more details.](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html)
  
- In this repo, there are 46 tags:
`['#', '$', "''", '(', ')', ',', '--s--', '.', ':', 'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '``']`

- There are 4 vocab txt files to build vocabulary for HMM independently:
  - ``WordList_TOEFL.txt``: Around 5.5k English words in TOEFL.
  - ``hmm_vocab.txt``: Around 23k English words.
  - ``COCA60000_vocab.txt``: Around 60k English words.
  - ``vocab.txt``: Around 600k English words.

## Implementation

### Create conda virtual environment and install packages

Create a new environment, for example ``postag``.

```sh
conda create -n postag python=3.10.12
conda activate postag
pip install -r requirement.txt
```

### Training and Testing
- Training and testing are executed in ``main.py``. Run this command to see argument parser:
  
```sh
python main.py -h
```
```sh
-------- ARGUMENT PARSER------------
usage: Viterbi POS-TAGGER 
--training_corpus: Training corpus. Choose ./data/WSJ_02-21.pos
--vocab_txt: Vocab txt file in data folder. There are 5k vocab
(./data/WordList_TOEFL.txt); 20k vocab (./data/hmm_vocab.txt);
60k vocab (./data/COCA60000_vocab.txt); 500k vocab
(./data/vocab.txt). Choose one of these.
--test_corpus: Choose one of these: ./data/WSJ_23.pos, ./data/WSJ_24.pos
--alpha:
Laplacian smoothing coefficient of HMM model. Alpha = 0.001 (default)

options:
  -h, --help            show this help message and exit
  --training_corpus TRAINING_CORPUS Training corpus .pos file, please Enter this: ./data/WSJ_02-21.pos
  --vocab_txt VOCAB_TXT Vocab txt file in data folder. Choose one of these (./data/WordList_TOEFL.txt, ./data/hmm_vocab.txt,
                        ./data/COCA60000_vocab.txt, ./data/vocab.txt)
  --test_corpus TEST_CORPUS Testing corpus .pos file
  --alpha ALPHA         Laplacian Smoothing Coefficient of HMM model
```
- Config your desired training_corpus, vocab_txt, test_corpus, alpha following above instruction. Then run the command line:

```sh
python main.py --training_corpus [training_corpus] --vocab_txt [vocab_txt] --test_corpus [test_corpus] --alpha [alpha]
```

For example:
```sh
python main.py --training_corpus './data/WSJ_02-21.pos' --vocab_txt './data/hmm_vocab.txt' --test_corpus './data/WSJ_24.pos' --alpha 0.001
```
## Inference

Getting tags of your new sentence is executed in ``predict_new.py``.

```sh
python predict_new.py -training_corpus [training_corpus] --vocab_txt [vocab_txt] --alpha [alpha]--stn [Your sentence]
```
For example:

```sh
python predict_new.py --training_corpus ./data/WSJ_02-21.pos --vocab_txt ./data/hmm_vocab.txt --alpha 0.001
--stn "When I see you face , there is not a thing that I could change . Because you are amazing , just the way you are." 
```

Result:

```sh
-------- YOUR SENTENCE TAGGING IS PROCESSING------------
-------Received parser inputs completed-----------
Sentence: ['When', 'I', 'see', 'you', 'face', ',', 'there', 'is', 'not', 'a', 'thing', 'that', 'I', 'could', 'change', '.', 'Because', 'you', 'are', 'amazing', ',', 'just', 'the', 'way', 'you', 'are', '.']
POS tags: ['WRB', 'PRP', 'VBP', 'PRP', 'VBP', ',', 'EX', 'VBZ', 'RB', 'DT', 'NN', 'IN', 'PRP', 'MD', 'VB', '.', 'IN', 'PRP', 'VBP', 'JJ', ',', 'RB', 'DT', 'NN', 'PRP', 'VBP', '.']
```

``Alternatively for Linux``: Train, test and getting tags for new sentence automatically by setting arguments in ``script.sh``:

```sh
#!/bin/bash


# You can adjust params then run bash script 

python main.py --training_corpus './data/WSJ_02-21.pos' --vocab_txt './data/hmm_vocab.txt' --test_corpus './data/WSJ_24.pos' --alpha 0.001

python predict_new.py --training_corpus "./data/WSJ_02-21.pos" --vocab_txt "./data/hmm_vocab.txt" --alpha 0.001 --stn "You just want attention. You dont want my heart."
```
Then, run this bash shell file:

```sh
bash script.sh
```

