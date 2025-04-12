from collections import Counter
import numpy as np

def subsample_frequent_words(tokens, threshold=1e-5):
    """
    Applique le sous-échantillonnage pour réduire la fréquence des mots fréquents.

    Args:
        tokens (list): Liste des tokens du corpus.
        threshold (float): Seuil pour le sous-échantillonnage (par défaut 1e-5).

    Returns:
        list: Liste des tokens après sous-échantillonnage.
    """
    # Calculer la fréquence des mots
    token_counts = Counter(tokens)
    total_tokens = sum(token_counts.values())

    # Calculer la probabilité de conservation pour chaque mot
    probabilities = {
        token: (np.sqrt(threshold / (count / total_tokens)) + 1) * (threshold / (count / total_tokens))
        for token, count in token_counts.items()
    }

    # Appliquer le sous-échantillonnage
    subsampled_tokens = [
        token for token in tokens if np.random.rand() < probabilities[token]
    ]

    return subsampled_tokens