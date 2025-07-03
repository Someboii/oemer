#!/usr/bin/env python3
"""
Percussion Voice Distribution Usage Examples
"""

import json
from oemer.percussion_distribution import (
    PercussionVoiceDistributor,
    MainInstrument,
    AccessoryInstrument,
    analyze_percussion_score
)


def example_basic_percussion_distribution():
    """Basic percussion distribution example"""
    print("=== Basic Percussion Distribution ===")
    
    # Example note events for a percussion score
    note_events = [
        # Bass drum notes
        {'bar_number': 1, 'beat_position': 1.0, 'pitch': 36, 'duration': 1.0, 'velocity': 100, 'note_id': 1, 'staff_line_pos': -4, 'track': 1},
        {'bar_number': 1, 'beat_position': 3.0, 'pitch': 36, 'duration': 1.0, 'velocity': 100, 'note_id': 2, 'staff_line_pos': -4, 'track': 1},
        
        # Snare drum notes
        {'bar_number': 1, 'beat_position': 2.0, 'pitch': 38, 'duration': 0.5, 'velocity': 90, 'note_id': 3, 'staff_line_pos': 0, 'track': 1},
        {'bar_number': 1, 'beat_position': 4.0, 'pitch': 38, 'duration': 0.5, 'velocity': 90, 'note_id': 4, 'staff_line_pos': 0, 'track': 1},
        
        # Cymbals
        {'bar_number': 2, 'beat_position': 1.0, 'pitch': 49, 'duration': 2.0, 'velocity': 110, 'note_id': 5, 'staff_line_pos': 6, 'track': 1},
        
        # Triangle (accessory)
        {'bar_number': 2, 'beat_position': 2.5, 'pitch': 81, 'duration': 0.25, 'velocity': 70, 'note_id': 6, 'staff_line_pos': 8, 'track': 1},
        {'bar_number': 2, 'beat_position': 3.0, 'pitch': 81, 'duration': 0.25, 'velocity': 70, 'note_id': 7, 'staff_line_pos': 8, 'track': 1},
        
        # Tambourine (accessory)
        {'bar_number': 1, 'beat_position': 1.5, 'pitch': 54, 'duration': 0.25, 'velocity': 80, 'note_id': 8, 'staff_line_pos': 4, 'track': 1},
        {'bar_number': 1, 'beat_position': 2.5, 'pitch': 54, 'duration': 0.25, 'velocity': 80, 'note_id': 9, 'staff_line_pos': 4, 'track': 1},
    ]
    
    # Initialize distributor for 3 players
    distributor = PercussionVoiceDistributor(num_players=3)
    
    # Distribute voices
    assignments = distributor.distribute_voices(note_events)
    
    # Print results
    print(f"Distribution for {distributor.num_players} players:")
    print(f"Total assignments: {len(assignments)}")
    
    for assignment in assignments:
        main_inst = assignment.main_instrument.value if assignment.main_instrument else "None"
        acc_insts = [inst.value for inst in assignment.accessory_instruments]
        
        print(f"\nPlayer {assignment.player_id}:")
        print(f"  Main instrument: {main_inst}")
        print(f"  Accessory instruments: {acc_insts}")
        print(f"  Total notes: {assignment.total_notes}")
        print(f"  Complexity score: {assignment.complexity_score:.3f}")
        
        # Show first few notes
        for i, note in enumerate(assignment.notes[:3]):
            print(f"    Note {i+1}: Bar {note.bar_number}, Beat {note.beat_position}, {note.instrument}")


def example_main_vs_accessory_rules():
    """Demonstrate main vs accessory instrument rules"""
    print("\n=== Main vs Accessory Instrument Rules ===")
    
    print("Main Instruments (should be played by different players):")
    for instrument in MainInstrument:
        print(f"  - {instrument.value}")
    
    print("\nAccessory Instruments (can be shared among players):")
    for instrument in AccessoryInstrument:
        print(f"  - {instrument.value}")
    
    print("\nDistribution Philosophy:")
    print("  1. Main instruments are assigned to different players when possible")
    print("  2. Each player should ideally have only one main instrument")
    print("  3. Accessory instruments can be distributed among multiple players")
    print("  4. Accessories are assigned based on timing conflicts and workload balance")


def example_complex_score():
    """Example with a more complex percussion score"""
    print("\n=== Complex Percussion Score Example ===")
    
    # Simulate a complex score with multiple main and accessory instruments
    note_events = []
    
    # Bass drum part (Player 1 main)
    for bar in range(1, 5):
        note_events.extend([
            {'bar_number': bar, 'beat_position': 1.0, 'pitch': 36, 'duration': 1.0, 'velocity': 100, 'note_id': len(note_events), 'staff_line_pos': -4, 'track': 1},
            {'bar_number': bar, 'beat_position': 3.0, 'pitch': 36, 'duration': 1.0, 'velocity': 100, 'note_id': len(note_events), 'staff_line_pos': -4, 'track': 1},
        ])
    
    # Snare drum part (Player 2 main)
    for bar in range(1, 5):
        for beat in [2.0, 4.0]:
            note_events.append({
                'bar_number': bar, 'beat_position': beat, 'pitch': 38, 'duration': 0.5, 'velocity': 90, 
                'note_id': len(note_events), 'staff_line_pos': 0, 'track': 1
            })
    
    # Xylophone part (Player 3 main)
    xylophone_pitches = [72, 74, 76, 77, 79]
    for bar in range(1, 3):
        for i, pitch in enumerate(xylophone_pitches):
            note_events.append({
                'bar_number': bar, 'beat_position': 1.0 + i * 0.5, 'pitch': pitch, 'duration': 0.25, 'velocity': 80,
                'note_id': len(note_events), 'staff_line_pos': 8 + i, 'track': 0
            })
    
    # Triangle (accessory - can go to any player)
    for bar in range(2, 5):
        note_events.append({
            'bar_number': bar, 'beat_position': 2.5, 'pitch': 81, 'duration': 0.25, 'velocity': 70,
            'note_id': len(note_events), 'staff_line_pos': 10, 'track': 1
        })
    
    # Tambourine (accessory - can go to any player)
    for bar in range(1, 5):
        for beat in [1.5, 2.5, 3.5, 4.5]:
            note_events.append({
                'bar_number': bar, 'beat_position': beat, 'pitch': 54, 'duration': 0.125, 'velocity': 75,
                'note_id': len(note_events), 'staff_line_pos': 4, 'track': 1
            })
    
    # Distribute with 3 players
    distributor = PercussionVoiceDistributor(num_players=3)
    assignments = distributor.distribute_voices(note_events)
    
    # Show detailed results
    summary = distributor.get_distribution_summary()
    
    print(f"Complex Score Distribution Results:")
    print(f"  Total players: {summary['total_players']}")
    print(f"  Main instruments assigned: {summary['main_instruments_assigned']}")
    print(f"  Accessory instruments assigned: {summary['accessory_instruments_assigned']}")
    print(f"  Total notes: {summary['total_notes']}")
    
    for detail in summary['player_details']:
        print(f"\nPlayer {detail['player_id']}:")
        print(f"  Main: {detail['main_instrument'] or 'None'}")
        print(f"  Accessories: {detail['accessory_instruments']}")
        print(f"  Notes: {detail['note_count']}")
        print(f"  Complexity: {detail['complexity_score']:.3f}")


def example_custom_rules():
    """Example with custom distribution rules"""
    print("\n=== Custom Distribution Rules Example ===")
    
    # Create distributor with custom rules
    distributor = PercussionVoiceDistributor(num_players=2)
    
    # Set custom rules
    custom_rules = {
        'main_instrument_priority': True,
        'accessory_flexibility': True,
        'simultaneous_conflict_penalty': 10.0,  # Higher penalty for conflicts
        'player_balance_weight': 2.0,           # Stronger balance preference
        'complexity_balance_weight': 0.5        # Lower complexity weight
    }
    
    distributor.set_distribution_rules(custom_rules)
    
    print("Custom rules applied:")
    for rule, value in custom_rules.items():
        print(f"  {rule}: {value}")
    
    print("\nThese rules will:")
    print("  - Strongly avoid simultaneous note conflicts")
    print("  - Prioritize balanced workload distribution")
    print("  - Reduce emphasis on complexity balancing")


def main():
    """Run all percussion examples"""
    print("Percussion Voice Distribution Examples\n")
    
    example_basic_percussion_distribution()
    example_main_vs_accessory_rules()
    example_complex_score()
    example_custom_rules()
    
    print("\n" + "="*50)
    print("Examples complete!")
    print("\nTo use with real scores:")
    print("  python -m oemer.enhanced_ete score.jpg --num-players 3")


if __name__ == "__main__":
    main()