mport os
import numpy as np
import random
import tensorflow as tf

def sentence_to_batch(inputStr, corpus, maxlen):
    vocab = corpus.dictionary.word2idx
    lines = []
    lengths = []
    words = inputStr[:-1].lower().strip().split(" ")
    words = ['<sos>'] + words
    words += ['<eos>']
    unk_idx = vocab['<oov>']
    indices = [vocab[w] if w in vocab else unk_idx for w in words]
    zeros = (maxlen-len(indices)+1)*[0]
    for i in range(64):
        lines.append(indices + zeros)
        lengths.append(len(indices) - 1)
    
    source = [x[:-1] for x in lines]
    target = [x[1:] for x in lines]
    
    return source,target,lengths
	
def get_string(max_indices, corpus):
    sentences = []
    for idx in max_indices:
        # generated sentence
        words = [corpus.dictionary.idx2word[x] for x in idx]
        # truncate sentences to first occurrence of <eos>
        sentence = []
        for w in words:
            if w != '<eos>':
                sentence.append(w)
            else:
                break
        sentences.append(" ".join(sentence))
    return sentences
