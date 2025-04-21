import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class ANNClassifier:
    def __init__(self, embedding_matrix, word2idx, input_length=100,
                 hidden_units=128, dropout_rate=0.4, learning_rate=0.001):
        """
        Initialise le classifieur ANN avec les embeddings.

        :param embedding_matrix: matrice d'embeddings (shape: vocab_size x embedding_dim)
        :param word2idx: dictionnaire mot → index
        :param input_length: longueur max. des documents (nombre de mots pris en compte)
        :param hidden_units: nombre de neurones dans la 1ère couche cachée
        :param dropout_rate: taux de dropout
        :param learning_rate: taux d’apprentissage pour Adam
        """
        self.embedding_matrix = embedding_matrix
        self.word2idx = word2idx
        self.input_length = input_length
        self.embedding_dim = embedding_matrix.shape[1]
        self.model = self._build_model(hidden_units, dropout_rate, learning_rate)

    def _build_model(self, hidden_units, dropout_rate, learning_rate):
        model = Sequential([
            Input(shape=(self.embedding_dim,)),

            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),

            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),

            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def _document_to_vector(self, document):
        vectors = []
        for word in document[:self.input_length]:
            idx = self.word2idx.get(word)
            if idx is not None and idx < self.embedding_matrix.shape[0]:
                vectors.append(self.embedding_matrix[idx])
        if not vectors:
            return np.zeros(self.embedding_dim)
        return np.mean(vectors, axis=0)

    def prepare_features(self, documents):
        return np.array([self._document_to_vector(doc) for doc in documents])

    def train(self, X, y, test_size=0.2, epochs=50, batch_size=32, verbose=1, patience=5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

        self.model.fit(X_train, y_train,
                       validation_data=(X_test, y_test),
                       epochs=epochs,
                       batch_size=batch_size,
                       verbose=verbose,
                       callbacks=[early_stop])

        y_pred = (self.model.predict(X_test) > 0.5).astype("int32")
        print("\nÉtiquettes réelles (y_test) :", y_test)
        print("Prédictions (y_pred) :", y_pred.flatten())

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

