import enum
import pickle
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from oemer import layers
from oemer.logger import get_logger
from oemer.bbox import get_center
from oemer.utils import get_unit_size

logger = get_logger(__name__)


class InstrumentType(enum.Enum):
    PIANO = "piano"
    VIOLIN = "violin"
    FLUTE = "flute"
    TRUMPET = "trumpet"
    DRUMS = "drums"
    GUITAR = "guitar"
    CELLO = "cello"
    CLARINET = "clarinet"
    SAXOPHONE = "saxophone"
    UNKNOWN = "unknown"


@dataclass
class NoteEvent:
    """Represents a note event with timing and characteristics"""
    bar_number: int
    beat_position: float
    pitch: int
    duration: float
    velocity: int
    instrument: InstrumentType
    track: int
    note_id: int


@dataclass
class PercussionVoice:
    """Represents a percussion voice assignment"""
    voice_id: int
    player_id: int
    note_events: List[NoteEvent]
    complexity_score: float


class InstrumentRecognizer:
    """Machine learning model to recognize instruments from musical notation"""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        self.feature_names = [
            'clef_type', 'key_signature', 'time_signature', 'note_range',
            'avg_note_duration', 'rhythm_complexity', 'articulation_marks',
            'dynamic_markings', 'staff_count', 'chord_frequency'
        ]
    
    def extract_features(self, score_data: Dict) -> np.ndarray:
        """Extract features from musical score for instrument classification"""
        features = []
        
        # Clef type (0=treble, 1=bass, 2=alto, 3=tenor)
        clef_type = self._get_clef_type(score_data)
        features.append(clef_type)
        
        # Key signature (number of sharps/flats)
        key_sig = self._get_key_signature(score_data)
        features.append(key_sig)
        
        # Time signature complexity
        time_sig = self._get_time_signature_complexity(score_data)
        features.append(time_sig)
        
        # Note range (highest - lowest pitch)
        note_range = self._get_note_range(score_data)
        features.append(note_range)
        
        # Average note duration
        avg_duration = self._get_average_duration(score_data)
        features.append(avg_duration)
        
        # Rhythm complexity score
        rhythm_complexity = self._get_rhythm_complexity(score_data)
        features.append(rhythm_complexity)
        
        # Articulation marks count
        articulation_count = self._get_articulation_count(score_data)
        features.append(articulation_count)
        
        # Dynamic markings count
        dynamic_count = self._get_dynamic_count(score_data)
        features.append(dynamic_count)
        
        # Staff count
        staff_count = self._get_staff_count(score_data)
        features.append(staff_count)
        
        # Chord frequency
        chord_freq = self._get_chord_frequency(score_data)
        features.append(chord_freq)
        
        return np.array(features).reshape(1, -1)
    
    def _get_clef_type(self, score_data: Dict) -> int:
        """Determine predominant clef type"""
        clefs = score_data.get('clefs', [])
        if not clefs:
            return 0
        
        clef_counts = defaultdict(int)
        for clef in clefs:
            clef_counts[clef.label.value] += 1
        
        return max(clef_counts.items(), key=lambda x: x[1])[0] if clef_counts else 0
    
    def _get_key_signature(self, score_data: Dict) -> int:
        """Get key signature complexity"""
        # This would be extracted from the key information
        return 0  # Placeholder
    
    def _get_time_signature_complexity(self, score_data: Dict) -> float:
        """Calculate time signature complexity"""
        # Complex time signatures get higher scores
        return 1.0  # Placeholder
    
    def _get_note_range(self, score_data: Dict) -> int:
        """Calculate the range of notes (highest - lowest pitch)"""
        notes = score_data.get('notes', [])
        if not notes:
            return 0
        
        pitches = [note.staff_line_pos for note in notes if hasattr(note, 'staff_line_pos')]
        if not pitches:
            return 0
        
        return max(pitches) - min(pitches)
    
    def _get_average_duration(self, score_data: Dict) -> float:
        """Calculate average note duration"""
        notes = score_data.get('notes', [])
        if not notes:
            return 0.0
        
        # This would calculate based on note types
        return 1.0  # Placeholder
    
    def _get_rhythm_complexity(self, score_data: Dict) -> float:
        """Calculate rhythm complexity score"""
        # More complex rhythms get higher scores
        return 1.0  # Placeholder
    
    def _get_articulation_count(self, score_data: Dict) -> int:
        """Count articulation marks"""
        return 0  # Placeholder
    
    def _get_dynamic_count(self, score_data: Dict) -> int:
        """Count dynamic markings"""
        return 0  # Placeholder
    
    def _get_staff_count(self, score_data: Dict) -> int:
        """Get number of staves"""
        staffs = score_data.get('staffs', [])
        return len(staffs) if staffs is not None else 1
    
    def _get_chord_frequency(self, score_data: Dict) -> float:
        """Calculate frequency of chords vs single notes"""
        groups = score_data.get('note_groups', [])
        if not groups:
            return 0.0
        
        chord_count = sum(1 for group in groups if len(group.note_ids) > 1)
        return chord_count / len(groups)
    
    def train(self, training_data: List[Tuple[Dict, InstrumentType]]):
        """Train the instrument recognition model"""
        X = []
        y = []
        
        for score_data, instrument in training_data:
            features = self.extract_features(score_data)
            X.append(features.flatten())
            y.append(instrument.value)
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Validate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Instrument recognition model trained with accuracy: {accuracy:.3f}")
        
        self.is_trained = True
    
    def predict(self, score_data: Dict) -> InstrumentType:
        """Predict instrument type from score data"""
        if not self.is_trained:
            logger.warning("Model not trained, returning UNKNOWN")
            return InstrumentType.UNKNOWN
        
        features = self.extract_features(score_data)
        prediction = self.model.predict(features)[0]
        
        try:
            return InstrumentType(prediction)
        except ValueError:
            return InstrumentType.UNKNOWN
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.is_trained = model_data['is_trained']
        self.feature_names = model_data['feature_names']


class NotePatternAnalyzer:
    """Analyzes note patterns and creates timing maps"""
    
    def __init__(self):
        self.note_events: List[NoteEvent] = []
        self.bar_patterns: Dict[int, List[NoteEvent]] = defaultdict(list)
        self.timing_map: Dict[float, List[NoteEvent]] = defaultdict(list)
    
    def analyze_score(self, score_data: Dict, instrument: InstrumentType) -> List[NoteEvent]:
        """Analyze the score and create note events"""
        notes = score_data.get('notes', [])
        groups = score_data.get('note_groups', [])
        measures = score_data.get('measures', {})
        
        note_events = []
        
        # Process each note group
        for group in groups:
            bar_number = self._get_bar_number(group, measures)
            beat_position = self._calculate_beat_position(group, measures.get(group.group, None))
            
            for note_id in group.note_ids:
                note = notes[note_id] if note_id < len(notes) else None
                if note and not note.invalid:
                    event = NoteEvent(
                        bar_number=bar_number,
                        beat_position=beat_position,
                        pitch=self._get_pitch(note),
                        duration=self._get_duration(note),
                        velocity=self._get_velocity(note),
                        instrument=instrument,
                        track=note.track,
                        note_id=note_id
                    )
                    note_events.append(event)
        
        self.note_events = note_events
        self._build_pattern_maps()
        return note_events
    
    def _get_bar_number(self, group, measures: Dict) -> int:
        """Determine which bar the note group belongs to"""
        # This would use the group's position and measure information
        return group.group if hasattr(group, 'group') else 1
    
    def _calculate_beat_position(self, group, measure) -> float:
        """Calculate the beat position within the measure"""
        # This would calculate based on the group's x_center and measure timing
        return 1.0  # Placeholder
    
    def _get_pitch(self, note) -> int:
        """Get MIDI pitch number from note"""
        # Convert staff line position to MIDI pitch
        base_pitch = 60  # Middle C
        return base_pitch + (note.staff_line_pos if hasattr(note, 'staff_line_pos') else 0)
    
    def _get_duration(self, note) -> float:
        """Get note duration in beats"""
        # Convert note type to duration
        duration_map = {
            'WHOLE': 4.0,
            'HALF': 2.0,
            'QUARTER': 1.0,
            'EIGHTH': 0.5,
            'SIXTEENTH': 0.25,
            'THIRTY_SECOND': 0.125,
            'SIXTY_FOURTH': 0.0625
        }
        
        if hasattr(note, 'label') and note.label:
            duration = duration_map.get(note.label.name, 1.0)
            if hasattr(note, 'has_dot') and note.has_dot:
                duration *= 1.5
            return duration
        return 1.0
    
    def _get_velocity(self, note) -> int:
        """Get note velocity (dynamics)"""
        # This would be extracted from dynamic markings
        return 80  # Default velocity
    
    def _build_pattern_maps(self):
        """Build pattern maps for analysis"""
        self.bar_patterns.clear()
        self.timing_map.clear()
        
        for event in self.note_events:
            self.bar_patterns[event.bar_number].append(event)
            timing_key = event.bar_number + event.beat_position
            self.timing_map[timing_key].append(event)
    
    def get_simultaneous_notes(self, tolerance: float = 0.1) -> Dict[float, List[NoteEvent]]:
        """Find notes that occur simultaneously within tolerance"""
        simultaneous = {}
        
        for timing, events in self.timing_map.items():
            if len(events) > 1:
                simultaneous[timing] = events
        
        return simultaneous
    
    def get_bar_complexity(self, bar_number: int) -> float:
        """Calculate complexity score for a bar"""
        events = self.bar_patterns.get(bar_number, [])
        if not events:
            return 0.0
        
        # Factors: note count, rhythm variety, pitch range
        note_count = len(events)
        unique_durations = len(set(event.duration for event in events))
        pitch_range = max(event.pitch for event in events) - min(event.pitch for event in events)
        
        complexity = (note_count * 0.4 + unique_durations * 0.3 + pitch_range * 0.3) / 10
        return min(complexity, 1.0)


class PercussionVoiceDistributor:
    """Distributes percussion voices based on note patterns and player availability"""
    
    def __init__(self, num_players: int):
        self.num_players = num_players
        self.voices: List[PercussionVoice] = []
        self.distribution_rules: Dict[str, float] = {
            'simultaneous_penalty': 2.0,
            'complexity_weight': 1.5,
            'hand_crossing_penalty': 1.8,
            'velocity_difference_weight': 0.8,
            'pitch_proximity_bonus': 0.6
        }
    
    def distribute_voices(self, note_events: List[NoteEvent], 
                         simultaneous_notes: Dict[float, List[NoteEvent]]) -> List[PercussionVoice]:
        """Distribute notes to percussion voices based on patterns and rules"""
        
        # Initialize voices
        self.voices = [
            PercussionVoice(
                voice_id=i,
                player_id=i % self.num_players,
                note_events=[],
                complexity_score=0.0
            )
            for i in range(self.num_players * 2)  # Allow multiple voices per player
        ]
        
        # Sort events by timing
        sorted_events = sorted(note_events, key=lambda e: (e.bar_number, e.beat_position))
        
        # Distribute events
        for event in sorted_events:
            best_voice = self._find_best_voice(event, simultaneous_notes)
            best_voice.note_events.append(event)
            best_voice.complexity_score = self._calculate_voice_complexity(best_voice)
        
        # Filter out empty voices
        self.voices = [voice for voice in self.voices if voice.note_events]
        
        return self.voices
    
    def _find_best_voice(self, event: NoteEvent, 
                        simultaneous_notes: Dict[float, List[NoteEvent]]) -> PercussionVoice:
        """Find the best voice for a note event"""
        timing_key = event.bar_number + event.beat_position
        simultaneous = simultaneous_notes.get(timing_key, [])
        
        best_voice = None
        best_score = float('inf')
        
        for voice in self.voices:
            score = self._calculate_assignment_score(event, voice, simultaneous)
            if score < best_score:
                best_score = score
                best_voice = voice
        
        return best_voice or self.voices[0]
    
    def _calculate_assignment_score(self, event: NoteEvent, voice: PercussionVoice,
                                  simultaneous: List[NoteEvent]) -> float:
        """Calculate score for assigning event to voice (lower is better)"""
        score = 0.0
        
        # Check for simultaneous note conflicts
        for sim_event in simultaneous:
            if any(e.note_id == sim_event.note_id for e in voice.note_events):
                score += self.distribution_rules['simultaneous_penalty']
        
        # Complexity penalty
        score += voice.complexity_score * self.distribution_rules['complexity_weight']
        
        # Hand crossing penalty (for percussion)
        if voice.note_events:
            last_event = voice.note_events[-1]
            pitch_diff = abs(event.pitch - last_event.pitch)
            if pitch_diff > 12:  # More than an octave
                score += self.distribution_rules['hand_crossing_penalty']
        
        # Velocity consistency bonus
        if voice.note_events:
            avg_velocity = sum(e.velocity for e in voice.note_events) / len(voice.note_events)
            velocity_diff = abs(event.velocity - avg_velocity)
            score += velocity_diff * self.distribution_rules['velocity_difference_weight'] / 127
        
        # Pitch proximity bonus
        if voice.note_events:
            recent_pitches = [e.pitch for e in voice.note_events[-3:]]
            min_pitch_diff = min(abs(event.pitch - p) for p in recent_pitches)
            score -= min_pitch_diff * self.distribution_rules['pitch_proximity_bonus'] / 12
        
        return score
    
    def _calculate_voice_complexity(self, voice: PercussionVoice) -> float:
        """Calculate complexity score for a voice"""
        if not voice.note_events:
            return 0.0
        
        events = voice.note_events
        
        # Factors: note density, rhythm variety, pitch range
        note_density = len(events) / max(1, events[-1].bar_number - events[0].bar_number + 1)
        unique_durations = len(set(e.duration for e in events))
        pitch_range = max(e.pitch for e in events) - min(e.pitch for e in events)
        
        complexity = (note_density * 0.4 + unique_durations * 0.3 + pitch_range * 0.3) / 10
        return min(complexity, 1.0)
    
    def set_distribution_rules(self, rules: Dict[str, float]):
        """Update distribution rules"""
        self.distribution_rules.update(rules)
    
    def get_voice_statistics(self) -> Dict[str, any]:
        """Get statistics about voice distribution"""
        if not self.voices:
            return {}
        
        stats = {
            'total_voices': len(self.voices),
            'total_notes': sum(len(voice.note_events) for voice in self.voices),
            'avg_complexity': sum(voice.complexity_score for voice in self.voices) / len(self.voices),
            'player_distribution': defaultdict(int)
        }
        
        for voice in self.voices:
            stats['player_distribution'][voice.player_id] += len(voice.note_events)
        
        return stats


def analyze_score_with_ml(score_data: Dict, num_players: int = 2) -> Dict[str, any]:
    """Main function to analyze score with ML and distribute voices"""
    
    # Initialize components
    instrument_recognizer = InstrumentRecognizer()
    pattern_analyzer = NotePatternAnalyzer()
    voice_distributor = PercussionVoiceDistributor(num_players)
    
    # Try to load pre-trained instrument model
    try:
        instrument_recognizer.load_model("oemer/models/instrument_classifier.pkl")
    except FileNotFoundError:
        logger.warning("No pre-trained instrument model found. Using default classification.")
    
    # Recognize instrument
    instrument = instrument_recognizer.predict(score_data)
    logger.info(f"Detected instrument: {instrument.value}")
    
    # Analyze note patterns
    note_events = pattern_analyzer.analyze_score(score_data, instrument)
    simultaneous_notes = pattern_analyzer.get_simultaneous_notes()
    
    # Distribute voices
    voices = voice_distributor.distribute_voices(note_events, simultaneous_notes)
    
    # Generate analysis report
    analysis_result = {
        'instrument': instrument,
        'note_events': note_events,
        'simultaneous_notes': simultaneous_notes,
        'voices': voices,
        'voice_statistics': voice_distributor.get_voice_statistics(),
        'bar_complexities': {
            bar: pattern_analyzer.get_bar_complexity(bar)
            for bar in pattern_analyzer.bar_patterns.keys()
        }
    }
    
    return analysis_result