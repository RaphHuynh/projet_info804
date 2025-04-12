import numpy as np
from src import TextPreprocessor
from src import CBOWModel
from src import SkipGramModel
from src import generate_skipgram_data
from src import generate_cbow_data
from src import plot_embeddings

# Corpus de texte
corpus = "Le roi aime la reine et la reine aime le roi. Le roi est puissant et la reine est belle. " \
         "Le roi et la reine se promènent dans le jardin. Ils aiment les fleurs et les oiseaux. " \
         "Le roi et la reine sont heureux ensemble. Ils aiment passer du temps ensemble. " \
         "Le roi et la reine sont les souverains du royaume. Ils gouvernent avec sagesse et bonté. " \
         "Le roi et la reine sont aimés de leur peuple. Ils organisent des fêtes et des célébrations. " \
         "Le roi et la reine sont un couple royal. Ils sont le symbole de l'amour et de l'harmonie. " \
         "Le roi et la reine sont unis par le destin. Ils sont liés par un amour éternel. " \
         "Le roi et la reine sont les gardiens de la paix. Ils protègent leur royaume des dangers. "

# Prétraitement
tp = TextPreprocessor(min_freq=1)
tokens = tp.preprocess(corpus)
tp.build_vocab(tokens)
encoded = tp.encode(tokens)

# Générer les données pour CBOW
X, y = generate_cbow_data(encoded, window=2)

# Générer les données pour SkipGram
X2 = generate_skipgram_data(encoded, window=2)

# Initialiser les labels pour SkipGram (1 pour chaque paire)
# Comme le SkipGram produit une paire (target, context) par token, les labels seront tous à 1
y2 = np.ones(len(X2[0]))  # Assumer que toutes les paires sont positives, donc label = 1

# Séparer les paires en target_words et context_words
target_words, context_words = X2[0], X2[1]  # X2 contient les paires (target, context)

# Vérification des dimensions
print(f"target_words shape: {target_words.shape}")
print(f"context_words shape: {context_words.shape}")
print(f"y2 shape: {y2.shape}")

# Initialisation du modèle CBOW
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

# Affichage des embeddings pour CBOW
embeddings = cbow.get_embeddings()
for word, idx in tp.word2idx.items():
    print(f"{word}: {embeddings[idx]}")

# Initialisation du modèle SkipGram
skipgram = SkipGramModel(
    vocab_size=len(tp.word2idx),
    embedding_dim=100,
    optimizer='sgd',
    learning_rate=0.01
)

skipgram.summary()

# Entraînement du modèle SkipGram
skipgram.train(target_words, context_words, y2, epochs=100)

# Affichage des embeddings pour SkipGram
embeddings = skipgram.get_embeddings()
for word, idx in tp.word2idx.items():
    print(f"{word}: {embeddings[idx]}")

# Affichage des embeddings via TSNE
plot_embeddings(embeddings, tp.idx2word, n_words=30, method='tsne')

