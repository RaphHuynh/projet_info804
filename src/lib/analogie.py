from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def analogie(motA, motB, motC, embeddings, word2idx, idx2word, top_n=5):
    if motA not in word2idx or motB not in word2idx or motC not in word2idx:
        return "Un des mots est absent du vocabulaire"

    vecA = embeddings[word2idx[motA]]
    vecB = embeddings[word2idx[motB]]
    vecC = embeddings[word2idx[motC]]

    # vecteur D = B - A + C
    target_vec = vecB - vecA + vecC

    # Calcul des similarités cosinus
    sims = cosine_similarity([target_vec], embeddings)[0]

    # Trier les indices par similarité décroissante
    sorted_indices = np.argsort(-sims)

    results = []
    for idx in sorted_indices:
        word = idx2word[idx]
        if word not in {motA, motB, motC}:
            results.append((word, sims[idx]))
        if len(results) >= top_n:
            break
    return results
