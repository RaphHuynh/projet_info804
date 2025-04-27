import numpy as np
from src import TextPreprocessor
from src import CBOWModel
from src import SkipGramModel
from src import generate_skipgram_data
from src import generate_cbow_data
from src import plot_embeddings
from src import generate_negative_samples
from src import subsample_frequent_words
from src import cosine_similarity

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
subsampled_tokens = subsample_frequent_words(tokens, threshold=1e-1)
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
positive_pairs = list(zip(X[0], X[1]))  # Paires positives (target, context)
negative_samples, labels = generate_negative_samples(len(tp.word2idx), positive_pairs, num_negative=5)

# Préparer les données pour CBOW
target_words, context_words = zip(*negative_samples)
target_words = np.array(target_words)

# Reformater context_words pour qu'il ait la bonne forme
num_samples = min(len(context_words), len(target_words))
context_words = np.array(context_words[:num_samples]).reshape(-1, cbow.context_size)
target_words = np.array(target_words[:num_samples])

# Vérifiez les formes des données
print(f"Forme de context_words : {context_words.shape}")
print(f"Forme de target_words : {target_words.shape}")

# Entraîner le modèle CBOW
cbow.train(context_words, target_words, epochs=100, batch_size=6)
cbow_embeddings = cbow.get_embeddings()

# Visualisation des embeddings du modèle CBOW
plot_embeddings(cbow_embeddings, tp.idx2word, n_words=30, method='tsne')

# Calcul de la matrice de similarité cosinus
cosine_similarity_matrix = cosine_similarity(cbow_embeddings)
print("Matrice de similarité cosinus (CBOW) :")
print(cosine_similarity_matrix)

# Afficher les mots les plus similaires pour CBOW
def print_most_similar_words(embeddings, word2idx, idx2word, cosine_similarity_matrix, top_n=5):
    for i in range(len(embeddings)):
        similar_indices = np.argsort(cosine_similarity_matrix[i])[-top_n:]
        similar_words = [idx2word[idx] for idx in similar_indices]
        print(f"Mot: {idx2word[i]}, Mots similaires: {similar_words}")

print_most_similar_words(cbow_embeddings, tp.word2idx, tp.idx2word, cosine_similarity_matrix, top_n=5)

# --- SkipGram avec Negative Sampling ---
# Générer les données pour SkipGram
X2 = generate_skipgram_data(encoded, window=2)

# Générer les exemples négatifs pour SkipGram
target_words, context_words = X2[0], X2[1]
positive_pairs = list(zip(target_words, context_words))
negative_samples, labels = generate_negative_samples(len(tp.word2idx), positive_pairs, num_negative=5)

# Préparer les données pour l'entraînement
target_words, context_words = zip(*negative_samples)
target_words = np.array(target_words)
context_words = np.array(context_words)
labels = np.array(labels)

# Initialiser et entraîner le modèle SkipGram avec échantillonnage négatif
skipgram = SkipGramModel(
    vocab_size=len(tp.word2idx),
    embedding_dim=100,
    optimizer='sgd',
    learning_rate=0.01
)
skipgram.summary()
skipgram.train(target_words, context_words, labels, epochs=100, batch_size=16)
skipgram_embeddings = skipgram.get_embeddings()

# Visualisation des embeddings du modèle SkipGram
plot_embeddings(skipgram_embeddings, tp.idx2word, n_words=30, method='pca')

# Calcul de la matrice de similarité cosinus
cosine_similarity_matrix = cosine_similarity(skipgram_embeddings)
print("Matrice de similarité cosinus (SkipGram) :")
print(cosine_similarity_matrix)

# Afficher les mots les plus similaires pour SkipGram
print_most_similar_words(skipgram_embeddings, tp.word2idx, tp.idx2word, cosine_similarity_matrix, top_n=5)