import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

import nltk

nltk.download('stopwords')

import numpy as np
from src import TextPreprocessor
from src import CBOWModel
from src import generate_cbow_data
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

print(f"Nombre de document avant prétraitement : {len(corpus)}")

corpus = corpus[:int(0.3 * len(corpus))]

print(f"Nombre de document apres prétraitement : {len(corpus)}")

corpus = " ".join(corpus)

tokens = tp.preprocess(corpus)
print(tokens[:10])
tp.build_vocab(tokens)

# Sous-échantillonnage des mots fréquents
subsampled_tokens = subsample_frequent_words(tokens, threshold=1e-4)
print(f"Nombre de tokens après sous-échantillonnage : {len(subsampled_tokens)}")

# Recalculer le vocabulaire à partir des tokens sous-échantillonnés
tp.build_vocab(subsampled_tokens)
encoded = tp.encode(subsampled_tokens)

# --- CBOW avec Negative Sampling ---
# Générer les données pour CBOW
X, y = generate_cbow_data(encoded, window=2)

# Initialiser et entraîner le modèle CBOW avec échantillonnage négatif
cbow = CBOWModel(
    vocab_size=len(tp.word2idx),
    embedding_dim=10,
    context_size=4,
    use_ann=True,
    optimizer='adam',
    learning_rate=0.01
)

# Générer les exemples négatifs pour CBOW
positive_pairs = []
for context, target in zip(X, y):
    for ctx_word in context:
        positive_pairs.append((target, ctx_word))
negative_samples, labels = generate_negative_samples(len(tp.word2idx), positive_pairs, num_negative=5)

# Préparer les données pour CBOW
target_words, context_words = zip(*negative_samples)
target_words = np.array(target_words)

# Reformater context_words pour qu'il ait la bonne forme
num_samples = min(len(context_words), len(target_words))
context_words = np.array(context_words[:num_samples]).reshape(-1, cbow.context_size)
target_words = np.array(target_words[:num_samples])

print(f"Nombre de paires positives : {len(positive_pairs)}")
print(f"Nombre de paires négatives : {len(negative_samples)}")

# Entraîner le modèle CBOW
cbow.train(context_words, target_words, epochs=100, batch_size=6)
cbow_embeddings = cbow.get_embeddings()

np.save("cbow_embeddings_30.npy", cbow_embeddings)