import numpy as np

def generate_cbow_data(encoded_text, window=2):
    X, y = [], []
    for i in range(window, len(encoded_text) - window):
        context = encoded_text[i - window:i] + encoded_text[i + 1:i + window + 1]
        target = encoded_text[i]
        X.append(context)
        y.append(target)
    return np.array(X), np.array(y)