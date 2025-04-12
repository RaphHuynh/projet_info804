# src/skipgram.py

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dot, Reshape, Activation
from tensorflow.keras.optimizers import Adam, SGD

class SkipGramModel:
    def __init__(self, vocab_size, embedding_dim=100, optimizer=None, learning_rate=0.01):
        """
        :param vocab_size: taille du vocabulaire
        :param embedding_dim: taille des vecteurs d'embedding
        :param optimizer: nom de l'optimiseur ('adam', 'sgd') ou None si on ne veut pas compiler tout de suite
        :param learning_rate: taux d'apprentissage
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer.lower() if optimizer else None
        self.model = self._build_model()

        if self.optimizer_name:
            self.compile_model(self.optimizer_name)

    def _build_model(self):
        input_word = Input(shape=(1,))
        context_word = Input(shape=(1,))
        
        embedding = Embedding(input_dim=self.vocab_size,
                              output_dim=self.embedding_dim,
                              input_length=1,
                              name="embedding")

        input_embedding = embedding(input_word)
        context_embedding = embedding(context_word)

        dot_product = Dot(axes=-1)([input_embedding, context_embedding])
        dot_product = Reshape((1,))(dot_product)
        output = Activation('sigmoid')(dot_product)

        return Model(inputs=[input_word, context_word], outputs=output)

    def compile_model(self, optimizer_name=None):
        """
        Compile le modèle avec l'optimiseur spécifié.
        """
        if optimizer_name:
            optimizer_name = optimizer_name.lower()
        elif self.optimizer_name:
            optimizer_name = self.optimizer_name
        else:
            raise ValueError("Aucun optimiseur spécifié. Passez-en un à compile_model(optimizer_name='adam')")

        if optimizer_name == 'adam':
            optimizer = Adam(learning_rate=self.learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = SGD(learning_rate=self.learning_rate)
        else:
            raise ValueError(f"Optimiseur non supporté: {optimizer_name}")

        self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    def summary(self):
        return self.model.summary()

    def train(self, target_words, context_words, labels, epochs=10, batch_size=512):
        self.model.fit([target_words, context_words], labels, epochs=epochs, batch_size=batch_size)

    def get_embeddings(self):
        return self.model.get_layer("embedding").get_weights()[0]