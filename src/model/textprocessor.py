import re
from collections import Counter
from nltk.corpus import stopwords

class TextPreprocessor:
    def __init__(self, min_freq=5):
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        self.stopwords = set(stopwords.words('english'))  # Chargé une seule fois

    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ\s]", "", text)
        tokens = text.split()
        tokens = [token for token in tokens if token not in self.stopwords]
        return tokens

    def build_vocab(self, tokens):
        freq = Counter(tokens)
        filtered = [word for word in tokens if freq[word] >= self.min_freq]
        self.vocab = set(filtered)
        self.word2idx = {word: idx for idx, word in enumerate(sorted(self.vocab))}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        return filtered

    def encode(self, tokens):
        return [self.word2idx[token] for token in tokens if token in self.word2idx]

    def decode(self, indices):
        return [self.idx2word[idx] for idx in indices]

