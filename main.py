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

# Corpus de texte
corpus = (
    "Le roi aime la reine et la reine aime le roi. Le roi est puissant et la reine est belle. "
    "Le roi et la reine se promènent dans le jardin. Ils aiment les fleurs et les oiseaux. "
    "Le roi et la reine sont heureux ensemble. Ils aiment passer du temps ensemble. "
    "Le roi et la reine sont les souverains du royaume. Ils gouvernent avec sagesse et bonté. "
    "Le roi et la reine sont aimés de leur peuple. Ils organisent des fêtes et des célébrations. "
    "Le roi et la reine sont un couple royal. Ils sont le symbole de l'amour et de l'harmonie. "
    "Le roi et la reine sont unis par le destin. Ils sont liés par un amour éternel. "
    "Le roi et la reine sont les gardiens de la paix. Ils protègent leur royaume des dangers. "
)

# Prétraitement
tp = TextPreprocessor(min_freq=1)
tokens = tp.preprocess(corpus)
tp.build_vocab(tokens)
encoded = tp.encode(tokens)

# Sous-échantillonnage des mots fréquents
subsampled_tokens = subsample_frequent_words(tokens, threshold=1e-5)
print(f"Nombre de tokens après sous-échantillonnage : {len(subsampled_tokens)}")

# Générer les données pour CBOW
X, y = generate_cbow_data(encoded, window=2)

# Générer les données pour SkipGram
X2 = generate_skipgram_data(encoded, window=2)

# Initialiser les labels pour SkipGram (1 pour chaque paire)
y2 = np.ones(len(X2[0]))  # toutes les paires sont positives
target_words, context_words = X2[0], X2[1]

# Vérification des dimensions
print(f"target_words shape: {target_words.shape}")
print(f"context_words shape: {context_words.shape}")
print(f"y2 shape: {y2.shape}")

# === CBOW ===
cbow = CBOWModel(
    vocab_size=len(tp.word2idx),
    embedding_dim=10,
    context_size=4,
    use_ann=True,
    optimizer='adam',
    learning_rate=0.01
)
cbow.summary()
cbow.train(X, y, epochs=100)
cbow_embeddings = cbow.get_embeddings()

# === SkipGram ===
skipgram = SkipGramModel(
    vocab_size=len(tp.word2idx),
    embedding_dim=100,
    optimizer='sgd',
    learning_rate=0.01
)
skipgram.summary()

# Générer les données pour SkipGram avec échantillonnage négatif
positive_pairs = list(zip(target_words, context_words))
negative_samples, labels = generate_negative_samples(len(tp.word2idx), positive_pairs, num_negative=5)

# Préparer les données pour l'entraînement
target_words, context_words = zip(*negative_samples)
target_words = np.array(target_words)
context_words = np.array(context_words)
labels = np.array(labels)

# Entraîner le modèle SkipGram avec échantillonnage négatif
skipgram.train(target_words, context_words, labels, epochs=100)
skipgram_embeddings = skipgram.get_embeddings()

# === Visualisation des embeddings (SkipGram ici) ===
plot_embeddings(skipgram_embeddings, tp.idx2word, n_words=30, method='tsne')

# === ANN Classifier ===
docs = [
    ["roi", "reine", "aime"],
    ["fleurs", "oiseaux", "jardin"],
    ["souverains", "royaume", "gouvernent"],
    ["fêtes", "célébrations", "heureux"],
    ["paix", "protection", "gardien"]
]
labels = np.array([1, 0, 1, 0, 1])  # exemple binaire

# Choix des embeddings à utiliser pour l'ANN
embeddings = cbow_embeddings  # ou skipgram_embeddings

# ANN training
ann = ANNClassifier(embedding_matrix=embeddings, word2idx=tp.word2idx)
X_feat = ann.prepare_features(docs)
ann.train(X_feat, labels)

# === Prédiction ===
test_docs = [
    ["roi", "reine"],
    ["fleurs", "jardin"],
    ["souverains", "royaume"]
]
X_test_feat = ann.prepare_features(test_docs)
predictions = ann.model.predict(X_test_feat)
predictions = (predictions > 0.5).astype("int32")

print("Prédictions sur les documents de test :")
print(predictions)

unique_predictions = np.unique(predictions)
print("Classes prédites :", unique_predictions)