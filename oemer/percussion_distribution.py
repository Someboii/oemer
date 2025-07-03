import enum
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
from oemer.logger import get_logger

logger = get_logger(__name__)


class MainInstrument(enum.Enum):
    """Main percussion instruments - should be played by different players"""
    BASS_DRUM = "bass_drum"
    CLASH_CYMBALS = "clash_cymbals"
    SNARE_DRUM = "snare_drum"
    TAM_TAM = "tam_tam"
    XYLOPHONE = "xylophone"
    GLOCKENSPIEL = "glockenspiel"
    MARIMBAPHONE = "marimbaphone"
    VIBRAPHONE = "vibraphone"


class AccessoryInstrument(enum.Enum):
    """Accessory instruments - can be distributed among players as needed"""
    TAMBOURINE = "tambourine"
    TRIANGLE = "triangle"
    WHIP = "whip"
    TOMTOMS = "tomtoms"
    WOODBLOCK = "woodblock"
    COWBELL = "cowbell"
    SHAKER = "shaker"
    CASTANETS = "castanets"
    CLAVES = "claves"
    GUIRO = "guiro"


@dataclass
class PercussionNote:
    """Represents a percussion note with instrument and timing"""
    bar_number: int
    beat_position: float
    instrument: str  # MainInstrument or AccessoryInstrument value
    duration: float
    velocity: int
    note_id: int
    is_main_instrument: bool


@dataclass
class PlayerAssignment:
    """Represents a player's instrument assignments"""
    player_id: int
    main_instrument: Optional[MainInstrument]
    accessory_instruments: Set[AccessoryInstrument]
    notes: List[PercussionNote]
    total_notes: int
    complexity_score: float


class PercussionInstrumentClassifier:
    """Classifies percussion instruments from musical notation"""
    
    def __init__(self):
        # Pitch ranges and characteristics for instrument identification
        self.instrument_characteristics = {
            # Main instruments
            MainInstrument.BASS_DRUM: {
                'pitch_range': (35, 40),  # Low pitches
                'staff_position': 'bottom',
                'note_head_type': 'x',
                'typical_dynamics': ['ff', 'f']
            },
            MainInstrument.SNARE_DRUM: {
                'pitch_range': (38, 42),
                'staff_position': 'middle',
                'note_head_type': 'x',
                'typical_dynamics': ['mf', 'f', 'ff']
            },
            MainInstrument.CLASH_CYMBALS: {
                'pitch_range': (49, 57),
                'staff_position': 'top',
                'note_head_type': 'x',
                'typical_dynamics': ['f', 'ff', 'fff']
            },
            MainInstrument.TAM_TAM: {
                'pitch_range': (45, 50),
                'staff_position': 'middle-high',
                'note_head_type': 'x',
                'typical_dynamics': ['p', 'pp', 'f', 'ff']
            },
            MainInstrument.XYLOPHONE: {
                'pitch_range': (65, 108),  # High pitched
                'staff_position': 'treble',
                'note_head_type': 'normal',
                'typical_dynamics': ['mp', 'mf', 'f']
            },
            MainInstrument.GLOCKENSPIEL: {
                'pitch_range': (79, 108),  # Very high pitched
                'staff_position': 'treble',
                'note_head_type': 'normal',
                'typical_dynamics': ['p', 'mp', 'mf']
            },
            MainInstrument.MARIMBAPHONE: {
                'pitch_range': (48, 84),  # Wide range, lower than xylophone
                'staff_position': 'treble-bass',
                'note_head_type': 'normal',
                'typical_dynamics': ['p', 'mp', 'mf']
            },
            MainInstrument.VIBRAPHONE: {
                'pitch_range': (53, 89),  # Mid-range
                'staff_position': 'treble',
                'note_head_type': 'normal',
                'typical_dynamics': ['p', 'mp', 'mf']
            },
            
            # Accessory instruments
            AccessoryInstrument.TAMBOURINE: {
                'pitch_range': (54, 54),  # Single pitch typically
                'staff_position': 'top',
                'note_head_type': 'x',
                'typical_dynamics': ['mp', 'mf', 'f']
            },
            AccessoryInstrument.TRIANGLE: {
                'pitch_range': (81, 81),  # High single pitch
                'staff_position': 'top',
                'note_head_type': 'diamond',
                'typical_dynamics': ['p', 'mp', 'mf']
            },
            AccessoryInstrument.WHIP: {
                'pitch_range': (76, 76),  # Single pitch
                'staff_position': 'middle-high',
                'note_head_type': 'x',
                'typical_dynamics': ['f', 'ff']
            },
            AccessoryInstrument.TOMTOMS: {
                'pitch_range': (41, 50),  # Range of tom pitches
                'staff_position': 'middle',
                'note_head_type': 'x',
                'typical_dynamics': ['mf', 'f']
            }
        }
    
    def classify_instrument(self, note_data: Dict) -> Tuple[str, bool]:
        """
        Classify a percussion instrument from note characteristics
        Returns: (instrument_name, is_main_instrument)
        """
        pitch = note_data.get('pitch', 60)
        staff_position = note_data.get('staff_position', 'middle')
        note_head_type = note_data.get('note_head_type', 'normal')
        dynamics = note_data.get('dynamics', 'mf')
        
        best_match = None
        best_score = 0
        
        # Check all instruments
        all_instruments = list(self.instrument_characteristics.keys())
        
        for instrument in all_instruments:
            characteristics = self.instrument_characteristics[instrument]
            score = 0
            
            # Pitch range matching
            pitch_range = characteristics['pitch_range']
            if pitch_range[0] <= pitch <= pitch_range[1]:
                score += 3
            elif abs(pitch - pitch_range[0]) <= 5 or abs(pitch - pitch_range[1]) <= 5:
                score += 1
            
            # Staff position matching
            if characteristics['staff_position'] == staff_position:
                score += 2
            
            # Note head type matching
            if characteristics['note_head_type'] == note_head_type:
                score += 2
            
            # Dynamics matching
            if dynamics in characteristics['typical_dynamics']:
                score += 1
            
            if score > best_score:
                best_score = score
                best_match = instrument
        
        if best_match:
            is_main = isinstance(best_match, MainInstrument)
            return best_match.value, is_main
        
        # Default fallback
        return AccessoryInstrument.TAMBOURINE.value, False


class PercussionVoiceDistributor:
    """Distributes percussion voices according to main/accessory instrument rules"""
    
    def __init__(self, num_players: int):
        self.num_players = num_players
        self.classifier = PercussionInstrumentClassifier()
        self.player_assignments: List[PlayerAssignment] = []
        
        # Distribution rules
        self.rules = {
            'main_instrument_priority': True,  # Main instruments get priority
            'accessory_flexibility': True,     # Accessories can be shared
            'simultaneous_conflict_penalty': 5.0,
            'player_balance_weight': 1.0,
            'complexity_balance_weight': 1.5
        }
    
    def distribute_voices(self, note_events: List[Dict]) -> List[PlayerAssignment]:
        """
        Distribute percussion voices according to main/accessory rules
        """
        # Step 1: Classify all notes
        classified_notes = self._classify_notes(note_events)
        
        # Step 2: Group by instrument
        instrument_groups = self._group_by_instrument(classified_notes)
        
        # Step 3: Assign main instruments to players
        main_assignments = self._assign_main_instruments(instrument_groups)
        
        # Step 4: Distribute accessory instruments
        final_assignments = self._distribute_accessories(main_assignments, instrument_groups)
        
        # Step 5: Optimize for conflicts and balance
        optimized_assignments = self._optimize_assignments(final_assignments)
        
        self.player_assignments = optimized_assignments
        return optimized_assignments
    
    def _classify_notes(self, note_events: List[Dict]) -> List[PercussionNote]:
        """Classify each note event as main or accessory instrument"""
        classified_notes = []
        
        for event in note_events:
            # Extract note characteristics for classification
            note_data = {
                'pitch': event.get('pitch', 60),
                'staff_position': self._determine_staff_position(event),
                'note_head_type': self._determine_note_head_type(event),
                'dynamics': event.get('dynamics', 'mf')
            }
            
            instrument_name, is_main = self.classifier.classify_instrument(note_data)
            
            perc_note = PercussionNote(
                bar_number=event.get('bar_number', 1),
                beat_position=event.get('beat_position', 1.0),
                instrument=instrument_name,
                duration=event.get('duration', 1.0),
                velocity=event.get('velocity', 80),
                note_id=event.get('note_id', 0),
                is_main_instrument=is_main
            )
            
            classified_notes.append(perc_note)
        
        return classified_notes
    
    def _determine_staff_position(self, event: Dict) -> str:
        """Determine staff position from note data"""
        staff_line_pos = event.get('staff_line_pos', 0)
        
        if staff_line_pos > 6:
            return 'top'
        elif staff_line_pos > 2:
            return 'middle-high'
        elif staff_line_pos > -2:
            return 'middle'
        else:
            return 'bottom'
    
    def _determine_note_head_type(self, event: Dict) -> str:
        """Determine note head type from note data"""
        # This would analyze the actual note head shape
        # For now, use simple heuristics
        track = event.get('track', 0)
        
        # Assume percussion tracks use 'x' note heads
        if track > 0:  # Non-melodic tracks
            return 'x'
        return 'normal'
    
    def _group_by_instrument(self, notes: List[PercussionNote]) -> Dict[str, List[PercussionNote]]:
        """Group notes by instrument type"""
        groups = defaultdict(list)
        
        for note in notes:
            groups[note.instrument].append(note)
        
        return dict(groups)
    
    def _assign_main_instruments(self, instrument_groups: Dict[str, List[PercussionNote]]) -> List[PlayerAssignment]:
        """Assign main instruments to different players"""
        assignments = []
        
        # Get all main instruments present in the score
        main_instruments_present = []
        for instrument_name, notes in instrument_groups.items():
            if notes and notes[0].is_main_instrument:
                try:
                    main_inst = MainInstrument(instrument_name)
                    main_instruments_present.append((main_inst, notes))
                except ValueError:
                    continue
        
        # Sort by number of notes (busiest instruments first)
        main_instruments_present.sort(key=lambda x: len(x[1]), reverse=True)
        
        # Assign to players
        for i, (main_instrument, notes) in enumerate(main_instruments_present):
            if i < self.num_players:
                player_id = i
            else:
                # More main instruments than players - assign to least busy player
                player_id = self._find_least_busy_player(assignments)
            
            # Find existing assignment or create new one
            assignment = self._find_or_create_assignment(assignments, player_id)
            
            if assignment.main_instrument is None:
                assignment.main_instrument = main_instrument
                assignment.notes.extend(notes)
            else:
                # Player already has a main instrument - this is a conflict
                logger.warning(f"Player {player_id} assigned multiple main instruments: "
                             f"{assignment.main_instrument.value} and {main_instrument.value}")
                # Assign to next available player or create overflow
                overflow_player = len(assignments)
                new_assignment = PlayerAssignment(
                    player_id=overflow_player,
                    main_instrument=main_instrument,
                    accessory_instruments=set(),
                    notes=notes.copy(),
                    total_notes=len(notes),
                    complexity_score=self._calculate_complexity(notes)
                )
                assignments.append(new_assignment)
        
        return assignments
    
    def _distribute_accessories(self, main_assignments: List[PlayerAssignment], 
                              instrument_groups: Dict[str, List[PercussionNote]]) -> List[PlayerAssignment]:
        """Distribute accessory instruments among players"""
        
        # Get all accessory instruments
        accessory_instruments = []
        for instrument_name, notes in instrument_groups.items():
            if notes and not notes[0].is_main_instrument:
                try:
                    acc_inst = AccessoryInstrument(instrument_name)
                    accessory_instruments.append((acc_inst, notes))
                except ValueError:
                    continue
        
        # Sort by complexity and timing conflicts
        accessory_instruments.sort(key=lambda x: (
            len(x[1]),  # Number of notes
            self._calculate_timing_conflicts(x[1], main_assignments)  # Timing conflicts
        ))
        
        assignments = main_assignments.copy()
        
        for accessory_instrument, notes in accessory_instruments:
            # Find best player for this accessory instrument
            best_player = self._find_best_player_for_accessory(
                accessory_instrument, notes, assignments
            )
            
            if best_player is not None:
                assignment = self._find_or_create_assignment(assignments, best_player)
                assignment.accessory_instruments.add(accessory_instrument)
                assignment.notes.extend(notes)
            else:
                # Create new player if needed
                new_player_id = len(assignments)
                new_assignment = PlayerAssignment(
                    player_id=new_player_id,
                    main_instrument=None,
                    accessory_instruments={accessory_instrument},
                    notes=notes.copy(),
                    total_notes=len(notes),
                    complexity_score=self._calculate_complexity(notes)
                )
                assignments.append(new_assignment)
        
        return assignments
    
    def _find_best_player_for_accessory(self, accessory_instrument: AccessoryInstrument, 
                                      notes: List[PercussionNote], 
                                      assignments: List[PlayerAssignment]) -> Optional[int]:
        """Find the best player for an accessory instrument"""
        
        if not assignments:
            return 0
        
        best_player = None
        best_score = float('inf')
        
        for assignment in assignments:
            score = 0
            
            # Check for timing conflicts
            conflicts = self._count_timing_conflicts(notes, assignment.notes)
            score += conflicts * self.rules['simultaneous_conflict_penalty']
            
            # Balance workload
            current_load = len(assignment.notes)
            score += current_load * self.rules['player_balance_weight']
            
            # Complexity balance
            score += assignment.complexity_score * self.rules['complexity_balance_weight']
            
            # Prefer players without main instruments for accessories
            if assignment.main_instrument is None:
                score -= 2.0
            
            if score < best_score:
                best_score = score
                best_player = assignment.player_id
        
        return best_player
    
    def _optimize_assignments(self, assignments: List[PlayerAssignment]) -> List[PlayerAssignment]:
        """Optimize assignments to reduce conflicts and balance workload"""
        
        # Update statistics for all assignments
        for assignment in assignments:
            assignment.total_notes = len(assignment.notes)
            assignment.complexity_score = self._calculate_complexity(assignment.notes)
        
        # Sort notes within each assignment by timing
        for assignment in assignments:
            assignment.notes.sort(key=lambda n: (n.bar_number, n.beat_position))
        
        return assignments
    
    def _find_or_create_assignment(self, assignments: List[PlayerAssignment], 
                                 player_id: int) -> PlayerAssignment:
        """Find existing assignment for player or create new one"""
        for assignment in assignments:
            if assignment.player_id == player_id:
                return assignment
        
        # Create new assignment
        new_assignment = PlayerAssignment(
            player_id=player_id,
            main_instrument=None,
            accessory_instruments=set(),
            notes=[],
            total_notes=0,
            complexity_score=0.0
        )
        assignments.append(new_assignment)
        return new_assignment
    
    def _find_least_busy_player(self, assignments: List[PlayerAssignment]) -> int:
        """Find the player with the least workload"""
        if not assignments:
            return 0
        
        min_load = min(len(assignment.notes) for assignment in assignments)
        for assignment in assignments:
            if len(assignment.notes) == min_load:
                return assignment.player_id
        
        return 0
    
    def _calculate_complexity(self, notes: List[PercussionNote]) -> float:
        """Calculate complexity score for a list of notes"""
        if not notes:
            return 0.0
        
        # Factors: note density, rhythm variety, dynamic changes
        total_duration = max(n.bar_number for n in notes) - min(n.bar_number for n in notes) + 1
        note_density = len(notes) / max(total_duration, 1)
        
        unique_durations = len(set(n.duration for n in notes))
        velocity_range = max(n.velocity for n in notes) - min(n.velocity for n in notes)
        
        complexity = (note_density * 0.5 + unique_durations * 0.3 + velocity_range * 0.2 / 127)
        return min(complexity, 1.0)
    
    def _calculate_timing_conflicts(self, notes: List[PercussionNote], 
                                  assignments: List[PlayerAssignment]) -> int:
        """Calculate timing conflicts between notes and existing assignments"""
        conflicts = 0
        
        for assignment in assignments:
            conflicts += self._count_timing_conflicts(notes, assignment.notes)
        
        return conflicts
    
    def _count_timing_conflicts(self, notes1: List[PercussionNote], 
                              notes2: List[PercussionNote]) -> int:
        """Count simultaneous notes between two note lists"""
        conflicts = 0
        tolerance = 0.1  # Beat position tolerance
        
        for note1 in notes1:
            for note2 in notes2:
                if (note1.bar_number == note2.bar_number and 
                    abs(note1.beat_position - note2.beat_position) < tolerance):
                    conflicts += 1
        
        return conflicts
    
    def get_distribution_summary(self) -> Dict[str, any]:
        """Get summary of voice distribution"""
        if not self.player_assignments:
            return {}
        
        summary = {
            'total_players': len(self.player_assignments),
            'main_instruments_assigned': 0,
            'accessory_instruments_assigned': 0,
            'total_notes': 0,
            'player_details': []
        }
        
        for assignment in self.player_assignments:
            if assignment.main_instrument:
                summary['main_instruments_assigned'] += 1
            
            summary['accessory_instruments_assigned'] += len(assignment.accessory_instruments)
            summary['total_notes'] += assignment.total_notes
            
            player_detail = {
                'player_id': assignment.player_id,
                'main_instrument': assignment.main_instrument.value if assignment.main_instrument else None,
                'accessory_instruments': [inst.value for inst in assignment.accessory_instruments],
                'note_count': assignment.total_notes,
                'complexity_score': assignment.complexity_score
            }
            summary['player_details'].append(player_detail)
        
        return summary
    
    def set_distribution_rules(self, rules: Dict[str, any]):
        """Update distribution rules"""
        self.rules.update(rules)


def analyze_percussion_score(score_data: Dict, num_players: int = 2) -> Dict[str, any]:
    """
    Main function to analyze percussion score and distribute voices
    """
    
    # Extract note events from score data
    note_events = []
    
    # Process note groups and notes
    groups = score_data.get('note_groups', [])
    notes = score_data.get('notes', [])
    
    for group in groups:
        for note_id in group.note_ids:
            if note_id < len(notes):
                note = notes[note_id]
                if not note.invalid:
                    event = {
                        'bar_number': getattr(group, 'group', 1),
                        'beat_position': 1.0,  # Would be calculated from timing
                        'pitch': getattr(note, 'staff_line_pos', 0) + 60,
                        'duration': 1.0,  # Would be extracted from note type
                        'velocity': 80,  # Default velocity
                        'note_id': note_id,
                        'staff_line_pos': getattr(note, 'staff_line_pos', 0),
                        'track': getattr(note, 'track', 0)
                    }
                    note_events.append(event)
    
    # Initialize distributor and analyze
    distributor = PercussionVoiceDistributor(num_players)
    assignments = distributor.distribute_voices(note_events)
    
    # Generate analysis results
    analysis_result = {
        'distributor': distributor,
        'assignments': assignments,
        'summary': distributor.get_distribution_summary(),
        'note_events': note_events,
        'classification_rules': {
            'main_instruments': [inst.value for inst in MainInstrument],
            'accessory_instruments': [inst.value for inst in AccessoryInstrument],
            'distribution_philosophy': {
                'main_instruments': 'Assigned to different players when possible',
                'accessory_instruments': 'Can be shared among players as needed'
            }
        }
    }
    
    return analysis_result