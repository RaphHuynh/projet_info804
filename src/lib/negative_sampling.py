import numpy as np

def generate_negative_samples(vocab_size, positive_pairs, num_negative=5):
    """
    Génère des exemples négatifs pour l'échantillonnage négatif.

    Args:
        vocab_size (int): Taille du vocabulaire.
        positive_pairs (list of tuples): Liste des paires positives (mot cible, mot contexte).
        num_negative (int): Nombre d'exemples négatifs à générer par paire positive.

    Returns:
        list of tuples: Liste des paires (mot cible, mot contexte) avec des exemples négatifs.
        list: Liste des labels (1 pour positif, 0 pour négatif).
    """
    negative_samples = []
    labels = []

    for target, context in positive_pairs:
        # Ajouter la paire positive
        negative_samples.append((target, context))
        labels.append(1)

        # Générer des exemples négatifs
        for _ in range(num_negative):
            negative_context = np.random.randint(0, vocab_size)
            while negative_context == target:  # Éviter que le mot négatif soit le même que le mot cible
                negative_context = np.random.randint(0, vocab_size)
            negative_samples.append((target, negative_context))
            labels.append(0)

    return negative_samples, labels