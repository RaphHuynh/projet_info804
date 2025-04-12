import numpy as np

def generate_cbow_data(encoded_text, window=2):
    X, y = [], []
    for i in range(window, len(encoded_text) - window):
        context = encoded_text[i - window:i] + encoded_text[i + 1:i + window + 1]
        target = encoded_text[i]
        X.append(context)
        y.append(target)
    return np.array(X), np.array(y)

def generate_skipgram_data(encoded_tokens, window=2):
    """
    Génère des paires (target, context) pour le training Skip-gram sans échantillonnage négatif.

    :param encoded_tokens: liste des tokens encodés (ex : [1, 2, 3, 4])
    :param window: taille de la fenêtre de contexte
    :return: paires (targets, contexts) sous forme de tableau 2D
    """
    pairs = []

    for i, target in enumerate(encoded_tokens):
        # Crée une fenêtre aléatoire entre 1 et window
        window_size = np.random.randint(1, window + 1)
        start = max(i - window_size, 0)
        end = min(i + window_size + 1, len(encoded_tokens))

        for j in range(start, end):
            if i != j:
                context = encoded_tokens[j]
                pairs.append((target, context))

    # Retourner les paires sous forme de tableau 2D
    pairs = np.array(pairs)
    return pairs  # On retourne les paires sous forme de tableau 2D, pas de décomposition ici

