import numpy as np
from collections import Counter

# max vector length in X_train found to be 64 (with X_val/X_test to be 35), so safe to pad vectors to size 64
# set pad_len=-1 for no padding applied
def simple_tokenizer(X, pad_len=64):
    words = ' '.join(X).split()
    count_words = Counter(words)
    total_words = len(words)
    sorted_words = count_words.most_common(total_words)
    vocab_to_int = {w:i+1 for i, (w,c) in enumerate(sorted_words)}

    X_v = []
    for x in X:
        v = [vocab_to_int[w] for w in x.split()]

        if pad_len >= 0:
            if len(v) > pad_len:
                v = v[:pad_len]
            v = np.pad(v, (0, pad_len-len(v)), 'constant')
        X_v.append(v)
    return X_v
