import numpy as np

from src import TextPreprocessor
from src import CBOWModel
from src import generate_cbow_data
from src import plot_embeddings


corpus = "Le roi aime la reine et la reine aime le roi. Le roi est puissant et la reine est belle. " \
         "Le roi et la reine se promènent dans le jardin. Ils aiment les fleurs et les oiseaux. " \
         "Le roi et la reine sont heureux ensemble. Ils aiment passer du temps ensemble. " \
         "Le roi et la reine sont les souverains du royaume. Ils gouvernent avec sagesse et bonté. " \
         "Le roi et la reine sont aimés de leur peuple. Ils organisent des fêtes et des célébrations. " \
         "Le roi et la reine sont un couple royal. Ils sont le symbole de l'amour et de l'harmonie. " \
         "Le roi et la reine sont unis par le destin. Ils sont liés par un amour éternel. " \
         "Le roi et la reine sont les gardiens de la paix. Ils protègent leur royaume des dangers. " \

# Prétraitement
tp = TextPreprocessor(min_freq=1)
tokens = tp.preprocess(corpus)
tp.build_vocab(tokens)
encoded = tp.encode(tokens)

X, y = generate_cbow_data(encoded, window=2)

# Initialisation du modèle
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

# Affichage des embeddings
embeddings = cbow.get_embeddings()
for word, idx in tp.word2idx.items():
    print(f"{word}: {embeddings[idx]}")
    
plot_embeddings(embeddings, tp.idx2word, n_words=30, method='tsne')
