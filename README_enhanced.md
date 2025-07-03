# Enhanced Oemer - ML-Powered OMR with Instrument Recognition and Voice Distribution

An enhanced version of the Oemer (End-to-end OMR) system that includes machine learning-powered instrument recognition and intelligent percussion voice distribution.

## New Features

### 🎵 Instrument Recognition
- Machine learning model to automatically identify instruments from musical notation
- Supports piano, violin, flute, trumpet, drums, guitar, cello, clarinet, saxophone
- Extensible framework for adding new instruments

### 🥁 Intelligent Voice Distribution
- Automatic distribution of percussion voices based on note patterns
- Configurable number of players
- Smart conflict resolution for simultaneous notes
- Complexity-based voice assignment

### 📊 Advanced Analysis
- Bar-by-bar complexity analysis
- Note pattern recognition and timing maps
- Simultaneous note detection
- Voice statistics and distribution reports

## Installation

```bash
# Install enhanced version
pip install -r requirements_enhanced.txt

# Install the enhanced oemer
pip install -e .
```

## Usage

### Basic Usage

```bash
# Run enhanced OMR with default settings
python -m oemer.enhanced_ete path/to/score.jpg --num-players 2

# With custom output directory
python -m oemer.enhanced_ete path/to/score.jpg -o ./output/ --num-players 3

# Save processing cache for faster re-runs
python -m oemer.enhanced_ete path/to/score.jpg --save-cache --num-players 2
```

### Advanced Usage

```python
from oemer.enhanced_ete import enhanced_extract
from oemer.instrument_recognition import PercussionVoiceDistributor

# Configure custom distribution rules
distributor = PercussionVoiceDistributor(num_players=3)
custom_rules = {
    'simultaneous_penalty': 3.0,
    'complexity_weight': 1.2,
    'hand_crossing_penalty': 2.5,
    'velocity_difference_weight': 0.5,
    'pitch_proximity_bonus': 0.8
}
distributor.set_distribution_rules(custom_rules)
```

## Output Files

The enhanced system generates several output files:

1. **`score.musicxml`** - Standard MusicXML file
2. **`score_analysis.json`** - ML analysis results including:
   - Detected instrument type
   - Voice distribution details
   - Bar complexity scores
   - Note pattern statistics
3. **`score_analysis.png`** - Visual analysis overlay

## Configuration

### Distribution Rules

You can customize how voices are distributed by adjusting these parameters:

- **`simultaneous_penalty`** (default: 2.0) - Penalty for assigning simultaneous notes to the same voice
- **`complexity_weight`** (default: 1.5) - Weight for voice complexity in assignment decisions
- **`hand_crossing_penalty`** (default: 1.8) - Penalty for large pitch jumps within a voice
- **`velocity_difference_weight`** (default: 0.8) - Weight for velocity consistency
- **`pitch_proximity_bonus`** (default: 0.6) - Bonus for keeping similar pitches together

### Number of Players

Set the number of available players for percussion parts:

```bash
--num-players 4  # For 4 percussion players
```

The system will distribute voices optimally among the available players.

## Machine Learning Models

### Instrument Recognition

The instrument recognition model uses features extracted from the musical score:

- Clef types and positions
- Key signatures
- Time signatures
- Note ranges and patterns
- Rhythm complexity
- Chord frequency
- Staff configurations

### Voice Distribution

The voice distribution system considers:

- Temporal relationships between notes
- Pitch patterns and hand positions
- Rhythm complexity
- Player capabilities and limitations
- Performance practicality

## Training Custom Models

### Instrument Recognition

```python
from oemer.instrument_recognition import InstrumentRecognizer, InstrumentType

# Prepare training data
training_data = [
    (score_data_1, InstrumentType.PIANO),
    (score_data_2, InstrumentType.VIOLIN),
    # ... more examples
]

# Train model
recognizer = InstrumentRecognizer()
recognizer.train(training_data)
recognizer.save_model("custom_instrument_model.pkl")
```

### Voice Complexity Prediction

```python
from oemer.models.instrument_classifier import AdvancedVoiceDistributor

# Initialize advanced distributor
distributor = AdvancedVoiceDistributor(max_voices=8)

# Train with your data
distributor.train_complexity_model(note_sequences, complexity_scores)
distributor.train_assignment_model(note_features, voice_states, assignments)

# Save trained models
distributor.save_models("custom_voice_models")
```

## API Reference

### Main Functions

- **`enhanced_extract(args)`** - Main processing function
- **`analyze_score_with_ml(score_data, num_players)`** - ML analysis pipeline

### Classes

- **`InstrumentRecognizer`** - ML model for instrument classification
- **`NotePatternAnalyzer`** - Analyzes note patterns and timing
- **`PercussionVoiceDistributor`** - Distributes voices among players
- **`AdvancedVoiceDistributor`** - Neural network-based voice distribution

## Examples

See `examples/enhanced_usage.py` for comprehensive usage examples including:

- Basic processing workflow
- Custom distribution rules
- Model training
- Results analysis
- Pattern analysis

## Performance Considerations

- **Memory Usage**: Enhanced analysis requires additional memory for ML models
- **Processing Time**: ML analysis adds 10-30% to processing time
- **Model Loading**: First run downloads and loads pre-trained models
- **Caching**: Use `--save-cache` to speed up repeated processing

## Troubleshooting

### Common Issues

1. **Model Loading Errors**: Ensure TensorFlow is properly installed
2. **Memory Issues**: Reduce batch sizes or use smaller models for large scores
3. **Voice Distribution**: Adjust distribution rules if results are unsatisfactory

### Debug Mode

Enable detailed logging:

```bash
export LOG_LEVEL=debug
python -m oemer.enhanced_ete path/to/score.jpg --num-players 2
```

## Contributing

Contributions are welcome! Areas for improvement:

- Additional instrument types
- Better voice distribution algorithms
- Performance optimizations
- New analysis features

## License

Same as original Oemer project - MIT License.