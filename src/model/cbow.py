from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Lambda
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow.keras.backend as K
import numpy as np

def data_generator(X, y, batch_size, vocab_size):
    num_samples = len(X)
    while True:
        for offset in range(0, num_samples, batch_size):
            batch_X = X[offset:offset+batch_size]
            batch_y = y[offset:offset+batch_size]
            batch_y_cat = to_categorical(batch_y, num_classes=vocab_size)
            yield batch_X, batch_y_cat


class CBOWModel:
    def __init__(self, vocab_size, embedding_dim, context_size, use_ann=True, optimizer='adam', learning_rate=0.01):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_size = context_size
        self.use_ann = use_ann
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _get_optimizer(self):
        if self.optimizer_name.lower() == 'sgd':
            return SGD(learning_rate=self.learning_rate)
        elif self.optimizer_name.lower() == 'adam':
            return Adam(learning_rate=self.learning_rate)
        else:
            raise ValueError(f"Optimiseur inconnu : {self.optimizer_name}")

    def _build_model(self):
        inputs = Input(shape=(self.context_size,))
        x = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim)(inputs)
        x = Lambda(lambda x: K.mean(x, axis=1))(x)

        if self.use_ann:
            outputs = Dense(self.vocab_size, activation='softmax')(x)
        else:
            outputs = x 

        model = Model(inputs=inputs, outputs=outputs)

        if self.use_ann:
            model.compile(loss='categorical_crossentropy', optimizer=self._get_optimizer())

        return model

    def train(self, X, y, epochs=10, batch_size=128, verbose=1):
        if not self.use_ann:
            raise RuntimeError("Ce modèle ne peut pas être entraîné sans couche ANN (use_ann=False).")
        
        gen = data_generator(X, y, batch_size, self.vocab_size)
        steps_per_epoch = len(X) // batch_size

        self.model.fit(gen, epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=verbose)

    def get_embeddings(self):
        return self.model.get_layer('embedding').get_weights()[0]

    def summary(self):
        return self.model.summary()