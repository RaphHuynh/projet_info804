import nltk

nltk.download('stopwords')

import numpy as np
from src import TextPreprocessor
from src import CBOWModel
from src import SkipGramModel
from src import generate_skipgram_data
from src import generate_cbow_data
from src import plot_embeddings
from src import ANNClassifier
from src import generate_negative_samples
from src import subsample_frequent_words
from src import *
import csv
import re
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords

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

tp = TextPreprocessor(min_freq=10)

corpus = corpus[:int(0.1 * len(corpus))]

corpus = " ".join(corpus)

tokens = tp.preprocess(corpus)
print(tokens[:10])
tp.build_vocab(tokens)

word2idx = tp.word2idx

# === Prétraitement ===
def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)  # supprime les balises HTML
    text = re.sub(r'[^\w\s]', '', text)  # supprime la ponctuation
    return text.lower().split()

documents = []
labels = []
with open("data/IMDB Dataset.csv", newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)  # saute l'en-tête
    for row in reader:
        if len(row) != 2:
            continue
        review, label = row
        words = clean_text(review)
        documents.append(words)
        labels.append(1 if label.lower() == "positive" else 0)

print(f"Nombre de documents : {len(documents)}")
print(f"Exemple nettoyé : {documents[0][:10]}")

# === Chargement des embeddings Google Word2Vec ===
from gensim.models import KeyedVectors

print("Chargement des embeddings Google...")
model_path = "data/GoogleNews-vectors-negative300.bin.gz"
word2vec = KeyedVectors.load_word2vec_format(model_path, binary=True)

# Construction du vocabulaire et de la matrice d'embedding
embedding_dim = 300
word2idx = {"<PAD>": 0}
embedding_matrix = [np.zeros(embedding_dim)]

for word in word2vec.key_to_index:
    word2idx[word] = len(word2idx)
    embedding_matrix.append(word2vec[word])

embedding_matrix = np.array(embedding_matrix)

# === Entraînement / Test ===
classifier = ANNClassifier(embedding_matrix=embedding_matrix, word2idx=word2idx, input_length=200)

X = classifier.prepare_features(documents)
y = np.array(labels)

# Répartition 70/30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entraînement
classifier.model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Prédictions
y_pred = (classifier.model.predict(X_test) > 0.5).astype("int32")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))