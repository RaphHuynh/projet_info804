from sklearn.metrics.pairwise import cosine_similarity

def similarite_cosinus(mot1, mot2, embeddings, word2idx):
    if mot1 not in word2idx or mot2 not in word2idx:
        return f"Mot introuvable dans le vocabulaire : {mot1} ou {mot2}"
    
    vec1 = embeddings[word2idx[mot1]].reshape(1, -1)
    vec2 = embeddings[word2idx[mot2]].reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]