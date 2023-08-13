import numpy as np 
import string 
# Punctuation characters
punct = set(string.punctuation)

# load in the training corpus
with open("./data/WSJ_02-21.pos", 'r') as f:
    training_corpus = f.readlines()

print(f"A few items of the training corpus list")
print(training_corpus[0:50])

for word_tag in training_corpus[:5]:
    print(word_tag.split())


