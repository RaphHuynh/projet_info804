import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class ANNClassifier:
    def __init__(self, embedding_matrix, word2idx, input_length=3, hidden_units=64, dropout_rate=0.3):
        """
        Initialise le classifieur ANN avec les embeddings.

        :param embedding_matrix: matrice d'embeddings (shape: vocab_size x embedding_dim)
        :param word2idx: dictionnaire mot → index
        :param input_length: nombre de mots dans chaque document
        :param hidden_units: nombre de neurones dans la couche cachée
        :param dropout_rate: taux de dropout
        """
        self.embedding_matrix = embedding_matrix
        self.word2idx = word2idx
        self.input_length = input_length
        self.model = self._build_model(hidden_units, dropout_rate)

    def _build_model(self, hidden_units, dropout_rate):
        input_dim = self.embedding_matrix.shape[1] * self.input_length
        model = Sequential()
        model.add(Dense(hidden_units, activation='relu', input_shape=(input_dim,)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='sigmoid'))  # classification binaire

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def _document_to_vector(self, document):
        vectors = []
        for word in document:
            idx = self.word2idx.get(word)
            if idx is not None:
                vectors.append(self.embedding_matrix[idx])
            else:
                vectors.append(np.zeros(self.embedding_matrix.shape[1]))
        # Pad or truncate to fixed input length
        vectors = vectors[:self.input_length]
        while len(vectors) < self.input_length:
            vectors.append(np.zeros(self.embedding_matrix.shape[1]))
        return np.concatenate(vectors)

    def prepare_features(self, documents):
        return np.array([self._document_to_vector(doc) for doc in documents])

    def train(self, X, y, test_size=0.2, epochs=30, batch_size=4):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(X_test, y_test))

        y_pred = (self.model.predict(X_test) > 0.5).astype("int32")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
