import os
import glob
import pandas as pd
import numpy as np
from src import TextPreprocessor
from src import ANNClassifierMultiClass
from src import *
from sklearn.utils.class_weight import compute_class_weight
import re
import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

import nltk

nltk.download('stopwords')

# === Chargement des embeddings Google Word2Vec ===
from gensim.models import KeyedVectors

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
encoded = tp.encode(subsampled_tokens)


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

# 1. Charger les documents et diviser par phrase
def load_documents(base_path):
    documents = []
    labels = []
    
    for filepath in glob.glob(os.path.join(base_path, '*'), recursive=False):
        if filepath.endswith('.txt'):
            with open(filepath, 'r', encoding='latin-1') as f:
                text = f.read()
            
            # Diviser le texte en phrases (ici simple approche basée sur des ponctuations)
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
            
            label = os.path.basename(filepath).replace('.txt', '')  # nom du fichier sans .txt
            
            # Ajouter chaque phrase comme un document avec la même étiquette
            for sentence in sentences:
                documents.append(sentence)
                labels.append(label)
                
    return documents, labels

# Charger les documents
base_path = 'data/archive'
documents_raw, labels_raw = load_documents(base_path)

print(f"Nombre de documents chargés : {len(documents_raw)}")
print(f"Labels trouvés : {sorted(set(labels_raw))}")

# 2. Prétraitement
text_processor = TextPreprocessor(min_freq=5)

# Tokeniser tous les documents
documents_tokenized = [text_processor.preprocess(doc) for doc in documents_raw]

# Construire vocabulaire
all_tokens = [token for tokens in documents_tokenized for token in tokens]
text_processor.build_vocab(all_tokens)

# 3. Encoder les documents
documents_encoded = [text_processor.encode(tokens) for tokens in documents_tokenized]

# 4. Préparer X et y
X = documents_encoded
y_labels = sorted(list(set(labels_raw)))  # toutes les classes distinctes triées
label2idx = {label: idx for idx, label in enumerate(y_labels)}
y = [label2idx[label] for label in labels_raw]

print(f"Nombre de classes : {len(label2idx)}")
print(f"Mapping label -> index : {label2idx}")

# 5. Séparer les données en train, val, test avec un stratified split
# Diviser d'abord en train et test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Diviser ensuite l'ensemble train en train et validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, stratify=y_train, random_state=42
)

print(f"Nombre d'exemples dans l'ensemble d'entraînement : {len(X_train)}")
print(f"Nombre d'exemples dans l'ensemble de validation : {len(X_val)}")
print(f"Nombre d'exemples dans l'ensemble de test : {len(X_test)}")

# Instancier ton classifieur
classifier = ANNClassifierMultiClass(
    embedding_matrix, 
    text_processor.word2idx, 
    input_length=50, 
    nb_classes=len(label2idx)
)

# Transformer les données pour le classifieur
X_train_ready = classifier.prepare_features(X_train)
X_val_ready = classifier.prepare_features(X_val)
X_test_ready = classifier.prepare_features(X_test)

# Calculer les poids des classes
classes = np.unique(y)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=np.array(y_train))
class_weight_dict = dict(zip(classes, weights))

# Entraîner
classifier.train(X_train_ready, np.array(y_train), 
                 test_size=0.4, epochs=50, batch_size=32, patience=5, class_weight=class_weight_dict)
