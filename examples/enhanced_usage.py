#!/usr/bin/env python3
"""
Enhanced OMR Usage Examples
"""

import os
import json
from pathlib import Path

from oemer.enhanced_ete import enhanced_extract
from oemer.instrument_recognition import (
    InstrumentRecognizer, 
    NotePatternAnalyzer, 
    PercussionVoiceDistributor,
    InstrumentType
)


def example_basic_usage():
    """Basic usage example"""
    print("=== Basic Enhanced OMR Usage ===")
    
    # Simulate command line arguments
    class Args:
        img_path = "path/to/your/score.jpg"
        output_path = "./output/"
        use_tf = False
        save_cache = True
        without_deskew = False
        num_players = 2
        distribution_rules = None
    
    args = Args()
    
    # Process the score
    try:
        result = enhanced_extract(args)
        print(f"Processing complete!")
        print(f"Detected instrument: {result['ml_analysis']['instrument'].value}")
        print(f"Generated {len(result['ml_analysis']['voices'])} voices")
    except Exception as e:
        print(f"Error: {e}")


def example_custom_distribution_rules():
    """Example with custom distribution rules"""
    print("=== Custom Distribution Rules Example ===")
    
    # Create custom rules
    custom_rules = {
        'simultaneous_penalty': 3.0,      # Higher penalty for simultaneous notes
        'complexity_weight': 1.2,         # Lower complexity weight
        'hand_crossing_penalty': 2.5,     # Higher hand crossing penalty
        'velocity_difference_weight': 0.5, # Lower velocity difference weight
        'pitch_proximity_bonus': 0.8      # Higher pitch proximity bonus
    }
    
    # Initialize voice distributor with custom rules
    distributor = PercussionVoiceDistributor(num_players=3)
    distributor.set_distribution_rules(custom_rules)
    
    print("Custom rules applied:")
    for rule, value in custom_rules.items():
        print(f"  {rule}: {value}")


def example_instrument_training():
    """Example of training instrument recognition model"""
    print("=== Instrument Training Example ===")
    
    # Create training data (this would come from labeled scores)
    training_data = [
        # (score_data_dict, instrument_type)
        ({}, InstrumentType.PIANO),
        ({}, InstrumentType.VIOLIN),
        # ... more training examples
    ]
    
    # Initialize and train recognizer
    recognizer = InstrumentRecognizer()
    
    if training_data:  # Only train if we have data
        recognizer.train(training_data)
        recognizer.save_model("trained_instrument_model.pkl")
        print("Instrument recognition model trained and saved!")
    else:
        print("No training data available for instrument recognition")


def example_voice_analysis():
    """Example of analyzing voice distribution results"""
    print("=== Voice Analysis Example ===")
    
    # Load analysis results (this would come from enhanced_extract)
    analysis_file = "output/score_analysis.json"
    
    if os.path.exists(analysis_file):
        with open(analysis_file, 'r') as f:
            analysis = json.load(f)
        
        print(f"Analysis Results:")
        print(f"  Instrument: {analysis['instrument']}")
        print(f"  Total Notes: {analysis['total_notes']}")
        print(f"  Voice Count: {analysis['voice_count']}")
        print(f"  Simultaneous Events: {analysis['simultaneous_events']}")
        
        print(f"\nVoice Distribution:")
        for voice in analysis['voice_details']:
            print(f"  Voice {voice['voice_id']} (Player {voice['player_id']}): "
                  f"{voice['note_count']} notes, "
                  f"complexity: {voice['complexity_score']:.3f}")
        
        print(f"\nBar Complexities:")
        for bar, complexity in analysis['bar_complexities'].items():
            print(f"  Bar {bar}: {complexity:.3f}")
    else:
        print(f"Analysis file not found: {analysis_file}")


def example_pattern_analysis():
    """Example of detailed pattern analysis"""
    print("=== Pattern Analysis Example ===")
    
    # This would use real score data
    score_data = {
        'notes': [],
        'note_groups': [],
        'staffs': [],
        'clefs': [],
        'sfns': [],
        'rests': [],
        'barlines': [],
        'measures': {}
    }
    
    # Analyze patterns
    analyzer = NotePatternAnalyzer()
    note_events = analyzer.analyze_score(score_data, InstrumentType.PIANO)
    simultaneous_notes = analyzer.get_simultaneous_notes()
    
    print(f"Pattern Analysis Results:")
    print(f"  Total note events: {len(note_events)}")
    print(f"  Simultaneous note groups: {len(simultaneous_notes)}")
    
    # Analyze complexity by bar
    for bar_num in range(1, 5):  # First 4 bars
        complexity = analyzer.get_bar_complexity(bar_num)
        print(f"  Bar {bar_num} complexity: {complexity:.3f}")


def main():
    """Run all examples"""
    print("Enhanced OMR Examples\n")
    
    example_basic_usage()
    print()
    
    example_custom_distribution_rules()
    print()
    
    example_instrument_training()
    print()
    
    example_voice_analysis()
    print()
    
    example_pattern_analysis()
    print()
    
    print("Examples complete!")


if __name__ == "__main__":
    main()