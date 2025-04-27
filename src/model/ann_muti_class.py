import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report

class ANNClassifierMultiClass:
    def __init__(self, embedding_matrix, word2idx, input_length=100, nb_classes=3):
        self.embedding_matrix = embedding_matrix
        self.word2idx = word2idx
        self.input_length = input_length
        self.nb_classes = nb_classes
        self.model = self.build_model()

    def build_model(self):
        input_layer = Input(shape=(self.input_length,))
        embedding_layer = Embedding(
            input_dim=self.embedding_matrix.shape[0],
            output_dim=self.embedding_matrix.shape[1],
            weights=[self.embedding_matrix],
            trainable=False
        )(input_layer)
        x = GlobalAveragePooling1D()(embedding_layer)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.5)(x)
        output_layer = Dense(self.nb_classes, activation='softmax')(x)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model

    def prepare_features(self, X):
        # Remplacer les indices hors de portée par un indice valide
        max_idx = self.embedding_matrix.shape[0] - 1  # Le dernier indice valide
        X_processed = [[min(idx, max_idx) for idx in seq] for seq in X]
        return pad_sequences(X_processed, maxlen=self.input_length, padding='post', truncating='post')

    def train(self, X, y, test_size=0.2, epochs=50, batch_size=32, patience=5, verbose=1, class_weight=None):
        # Séparation des données en train, validation et test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Séparer une partie de l'ensemble d'entraînement pour la validation
        X_train_ready, X_val_ready, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

        # Calcul des poids des classes si non fournis
        if class_weight is None:
            classes = np.unique(y_train)
            weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
            class_weight = dict(zip(classes, weights))

        print(f"Class weights applied: {class_weight}")  # Pour visualiser les poids des classes

        # Préparation des données avec padding
        X_train_ready = self.prepare_features(X_train_ready)
        X_val_ready = self.prepare_features(X_val_ready)
        X_test_ready = self.prepare_features(X_test)

        # Entraînement du modèle avec EarlyStopping
        early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

        self.model.fit(
            X_train_ready,
            y_train,
            validation_data=(X_val_ready, y_val),  # Utilisation de X_val_ready pour la validation
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=[early_stop],  # Appel d'early stopping
            class_weight=class_weight  # Poids des classes
        )

        # Prédictions sur l'ensemble de test
        y_pred = np.argmax(self.model.predict(X_test_ready), axis=1)

        # Affichage des résultats
        print("\nÉtiquettes réelles (y_test) :", y_test)
        print("Prédictions (y_pred) :", y_pred)

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

    def evaluate(self, X, y):
        # Préparer les données de test
        X_ready = self.prepare_features(X)
        
        # Évaluation du modèle
        loss, accuracy = self.model.evaluate(X_ready, y)
        print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
        return loss, accuracy

    def predict(self, X):
        X_processed = self.prepare_features(X)
        predictions = self.model.predict(X_processed)
        return np.argmax(predictions, axis=1)
