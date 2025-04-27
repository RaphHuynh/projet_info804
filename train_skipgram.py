import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

import nltk

nltk.download('stopwords')

import numpy as np
from src import TextPreprocessor
from src import CBOWModel
from src import generate_skipgram_data
from src import plot_embeddings
from src import ANNClassifier
from src import generate_negative_samples
from src import subsample_frequent_words
from src import *

import os

def load_text_data(directory):
    """
    Charge tous les fichiers .txt dans un répertoire donné et retourne leur contenu sous forme de liste.
    
    Args:
        directory (str): Chemin vers le répertoire contenant les fichiers .txt.
    
    Returns:
        list: Liste des phrases extraites des fichiers .txt.
    """
    corpus = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='latin1') as file:
                corpus.append(file.read())
    return corpus

# Exemple d'utilisation
directory = "data/archive/"
corpus = load_text_data(directory)
print(f"Nombre de fichiers chargés : {len(corpus)}") # Affiche les 100 premiers caractères du premier fichie

tp = TextPreprocessor(min_freq=10)

corpus = corpus[:int(0.3 * len(corpus))]

corpus = " ".join(corpus)

tokens = tp.preprocess(corpus)
print(tokens[:10])
tp.build_vocab(tokens)

# Sous-échantillonnage des mots fréquents
subsampled_tokens = subsample_frequent_words(tokens, threshold=1e-4)
print(f"Nombre de tokens après sous-échantillonnage : {len(subsampled_tokens)}")
encoded = tp.encode(subsampled_tokens)

X2 = generate_skipgram_data(encoded, window=2)

skipgram = SkipGramModel(
    vocab_size=len(tp.word2idx),
    embedding_dim=100,
    optimizer='sgd',
    learning_rate=0.01
)

skipgram.summary()

# Étape 1 : tu génères toutes les paires target/context
positive_pairs = generate_skipgram_data(encoded, window=2)

# Étape 2 : génération des exemples négatifs
negative_samples, labels = generate_negative_samples(len(tp.word2idx), positive_pairs, num_negative=5)

# Étape 3 : décomposer en target/context
target_words, context_words = zip(*negative_samples)
target_words = np.array(target_words, dtype=np.int32)
context_words = np.array(context_words, dtype=np.int32)
labels = np.array(labels, dtype=np.int32)
print(f"target_words.shape: {target_words.shape}")
print(f"context_words.shape: {context_words.shape}")
print(f"labels.shape: {labels.shape}")

# Entraîner le modèle SkipGram avec échantillonnage négatif et sous-échantillonnages
skipgram.train(target_words, context_words, labels, epochs=100)

skipgram_embeddings = skipgram.get_embeddings()

np.save("embeddings_skipgram_30.npy", skipgram_embeddings)