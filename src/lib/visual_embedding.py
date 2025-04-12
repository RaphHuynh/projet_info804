import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

def plot_embeddings(embeddings, idx2word, n_words=50, method='tsne', perplexity=5):
    # Limite le nombre de mots à n_words
    words = list(idx2word.keys())[:n_words]
    vectors = np.array([embeddings[i] for i in words])

    # Assure-toi que la perplexité est plus petite que le nombre de vecteurs
    if perplexity >= len(vectors):
        print(f"Perplexité trop grande, la définissons à {len(vectors)-1}")
        perplexity = len(vectors) - 1

    # Réduction de la dimensionnalité avec TSNE ou PCA
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    elif method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
    else:
        raise ValueError("Méthode inconnue : utiliser 'tsne' ou 'pca'.")

    reduced = reducer.fit_transform(vectors)

    # Affichage des embeddings
    plt.figure(figsize=(10, 8))
    for i, word_idx in enumerate(words):
        plt.scatter(reduced[i, 0], reduced[i, 1])
        plt.annotate(idx2word[word_idx], (reduced[i, 0] + 0.01, reduced[i, 1] + 0.01))
    plt.title(f"Visualisation des embeddings ({method.upper()})")
    plt.grid(True)
    plt.show()
