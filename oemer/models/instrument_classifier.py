import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Tuple, List


def create_instrument_classifier(input_features: int = 10, num_instruments: int = 10) -> Model:
    """Create a neural network for instrument classification"""
    
    inputs = tf.keras.Input(shape=(input_features,), name='score_features')
    
    # Feature processing layers
    x = layers.Dense(64, activation='relu', name='feature_dense_1')(inputs)
    x = layers.BatchNormalization(name='batch_norm_1')(x)
    x = layers.Dropout(0.3, name='dropout_1')(x)
    
    x = layers.Dense(32, activation='relu', name='feature_dense_2')(x)
    x = layers.BatchNormalization(name='batch_norm_2')(x)
    x = layers.Dropout(0.2, name='dropout_2')(x)
    
    x = layers.Dense(16, activation='relu', name='feature_dense_3')(x)
    x = layers.Dropout(0.1, name='dropout_3')(x)
    
    # Output layer
    outputs = layers.Dense(num_instruments, activation='softmax', name='instrument_output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='instrument_classifier')
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_voice_complexity_predictor(sequence_length: int = 32) -> Model:
    """Create a model to predict voice complexity from note sequences"""
    
    # Note features: [pitch, duration, velocity, beat_position]
    note_input = tf.keras.Input(shape=(sequence_length, 4), name='note_sequence')
    
    # LSTM layers for sequence processing
    x = layers.LSTM(64, return_sequences=True, name='lstm_1')(note_input)
    x = layers.Dropout(0.2, name='lstm_dropout_1')(x)
    
    x = layers.LSTM(32, return_sequences=False, name='lstm_2')(x)
    x = layers.Dropout(0.2, name='lstm_dropout_2')(x)
    
    # Dense layers for complexity prediction
    x = layers.Dense(16, activation='relu', name='complexity_dense_1')(x)
    x = layers.Dropout(0.1, name='complexity_dropout')(x)
    
    # Output: complexity score (0-1)
    complexity_output = layers.Dense(1, activation='sigmoid', name='complexity_score')(x)
    
    model = Model(inputs=note_input, outputs=complexity_output, name='voice_complexity_predictor')
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model


def create_voice_assignment_network(max_voices: int = 8, feature_dim: int = 16) -> Model:
    """Create a network for optimal voice assignment"""
    
    # Input: note features and current voice states
    note_features = tf.keras.Input(shape=(feature_dim,), name='note_features')
    voice_states = tf.keras.Input(shape=(max_voices, feature_dim), name='voice_states')
    
    # Process note features
    note_processed = layers.Dense(32, activation='relu', name='note_processor')(note_features)
    note_processed = layers.BatchNormalization(name='note_batch_norm')(note_processed)
    
    # Process voice states
    voice_processed = layers.TimeDistributed(
        layers.Dense(32, activation='relu'), name='voice_processor'
    )(voice_states)
    voice_processed = layers.GlobalAveragePooling1D(name='voice_pooling')(voice_processed)
    
    # Combine features
    combined = layers.Concatenate(name='feature_combination')([note_processed, voice_processed])
    
    # Decision layers
    x = layers.Dense(64, activation='relu', name='decision_dense_1')(combined)
    x = layers.Dropout(0.3, name='decision_dropout_1')(x)
    
    x = layers.Dense(32, activation='relu', name='decision_dense_2')(x)
    x = layers.Dropout(0.2, name='decision_dropout_2')(x)
    
    # Output: probability distribution over voices
    voice_assignment = layers.Dense(max_voices, activation='softmax', name='voice_assignment')(x)
    
    model = Model(
        inputs=[note_features, voice_states], 
        outputs=voice_assignment, 
        name='voice_assignment_network'
    )
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


class AdvancedVoiceDistributor:
    """Advanced voice distributor using neural networks"""
    
    def __init__(self, max_voices: int = 8):
        self.max_voices = max_voices
        self.complexity_model = create_voice_complexity_predictor()
        self.assignment_model = create_voice_assignment_network(max_voices)
        
    def predict_complexity(self, note_sequence: np.ndarray) -> float:
        """Predict complexity score for a note sequence"""
        if len(note_sequence.shape) == 2:
            note_sequence = note_sequence.reshape(1, *note_sequence.shape)
        
        complexity = self.complexity_model.predict(note_sequence, verbose=0)
        return float(complexity[0, 0])
    
    def assign_voice(self, note_features: np.ndarray, voice_states: np.ndarray) -> int:
        """Assign a note to the best voice using the neural network"""
        if len(note_features.shape) == 1:
            note_features = note_features.reshape(1, -1)
        if len(voice_states.shape) == 2:
            voice_states = voice_states.reshape(1, *voice_states.shape)
        
        assignment_probs = self.assignment_model.predict(
            [note_features, voice_states], verbose=0
        )
        return int(np.argmax(assignment_probs[0]))
    
    def train_complexity_model(self, training_sequences: List[np.ndarray], 
                             complexity_scores: List[float], epochs: int = 50):
        """Train the complexity prediction model"""
        X = np.array(training_sequences)
        y = np.array(complexity_scores)
        
        self.complexity_model.fit(X, y, epochs=epochs, validation_split=0.2, verbose=1)
    
    def train_assignment_model(self, note_features: List[np.ndarray], 
                             voice_states: List[np.ndarray], 
                             assignments: List[int], epochs: int = 50):
        """Train the voice assignment model"""
        X_notes = np.array(note_features)
        X_voices = np.array(voice_states)
        y = tf.keras.utils.to_categorical(assignments, num_classes=self.max_voices)
        
        self.assignment_model.fit(
            [X_notes, X_voices], y, 
            epochs=epochs, validation_split=0.2, verbose=1
        )
    
    def save_models(self, base_path: str):
        """Save trained models"""
        self.complexity_model.save(f"{base_path}_complexity.h5")
        self.assignment_model.save(f"{base_path}_assignment.h5")
    
    def load_models(self, base_path: str):
        """Load trained models"""
        self.complexity_model = tf.keras.models.load_model(f"{base_path}_complexity.h5")
        self.assignment_model = tf.keras.models.load_model(f"{base_path}_assignment.h5")


if __name__ == "__main__":
    # Example usage
    instrument_model = create_instrument_classifier()
    complexity_model = create_voice_complexity_predictor()
    assignment_model = create_voice_assignment_network()
    
    print("Instrument Classifier:")
    print(instrument_model.summary())
    print("\nVoice Complexity Predictor:")
    print(complexity_model.summary())
    print("\nVoice Assignment Network:")
    print(assignment_model.summary())