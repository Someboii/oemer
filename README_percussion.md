# Enhanced Oemer - Percussion Voice Distribution

An enhanced version of Oemer that intelligently distributes percussion voices based on main and accessory instrument classification rules.

## Percussion Distribution Rules

### Main Instruments
These instruments should be played by **different players** when possible:

- **Bass Drum** - Low-pitched, powerful foundation
- **Clash Cymbals** - High-pitched, dramatic accents  
- **Snare Drum** - Mid-pitched, rhythmic backbone
- **Tam Tam** - Large gong, sustained tones
- **Xylophone** - High-pitched melodic percussion
- **Glockenspiel** - Very high-pitched, bell-like tones
- **Marimbaphone** - Wide range, warm wooden tones
- **Vibraphone** - Mid-range, sustained metallic tones

### Accessory Instruments
These instruments **can be shared** among players as needed:

- **Tambourine** - Rhythmic shaking/striking
- **Triangle** - High-pitched metallic accent
- **Whip** - Sharp percussive accent
- **Tomtoms** - Various pitched drums
- **Woodblock** - Sharp wooden sound
- **Cowbell** - Metallic rhythmic accent
- **Shaker** - Continuous rhythmic texture
- **Castanets** - Rapid clicking sounds
- **Claves** - Sharp wooden clicks
- **Guiro** - Scraped rhythmic texture

## Usage

### Basic Usage

```bash
# Distribute percussion parts among 2 players
python -m oemer.enhanced_ete percussion_score.jpg --num-players 2

# Distribute among 4 players for complex scores
python -m oemer.enhanced_ete percussion_score.jpg --num-players 4

# Save processing cache for faster re-runs
python -m oemer.enhanced_ete percussion_score.jpg --num-players 3 --save-cache
```

### Advanced Usage

```python
from oemer.percussion_distribution import PercussionVoiceDistributor, analyze_percussion_score

# Analyze a percussion score
score_data = {
    'notes': notes,
    'note_groups': groups,
    'staffs': staffs,
    # ... other score data
}

# Distribute voices among 3 players
analysis = analyze_percussion_score(score_data, num_players=3)

# Get distribution summary
summary = analysis['summary']
print(f"Main instruments assigned: {summary['main_instruments_assigned']}")
print(f"Accessory instruments assigned: {summary['accessory_instruments_assigned']}")
```

## Distribution Algorithm

### Step 1: Instrument Classification
- Analyzes pitch range, staff position, note head type, and dynamics
- Classifies each note as main or accessory instrument
- Uses characteristic patterns for each instrument type

### Step 2: Main Instrument Assignment
- Assigns main instruments to different players when possible
- Prioritizes busiest instruments first
- Creates overflow players if more main instruments than available players

### Step 3: Accessory Distribution
- Distributes accessory instruments based on:
  - Timing conflicts with existing assignments
  - Player workload balance
  - Complexity distribution
- Allows multiple accessories per player

### Step 4: Optimization
- Minimizes simultaneous note conflicts
- Balances workload across players
- Considers performance practicality

## Output Files

The system generates:

1. **`score.musicxml`** - Standard MusicXML file
2. **`score_percussion_analysis.json`** - Detailed analysis including:
   - Player assignments with instruments
   - Note distribution details
   - Complexity scores
   - Classification rules applied
3. **`score_analysis.png`** - Visual analysis overlay

## Example Output

```json
{
  "summary": {
    "total_players": 3,
    "main_instruments_assigned": 3,
    "accessory_instruments_assigned": 2,
    "total_notes": 45,
    "player_details": [
      {
        "player_id": 0,
        "main_instrument": "bass_drum",
        "accessory_instruments": ["tambourine"],
        "note_count": 12,
        "complexity_score": 0.65
      },
      {
        "player_id": 1,
        "main_instrument": "snare_drum",
        "accessory_instruments": [],
        "note_count": 18,
        "complexity_score": 0.82
      },
      {
        "player_id": 2,
        "main_instrument": "xylophone",
        "accessory_instruments": ["triangle"],
        "note_count": 15,
        "complexity_score": 0.73
      }
    ]
  }
}
```

## Configuration

### Distribution Rules

Customize the distribution behavior:

```python
distributor = PercussionVoiceDistributor(num_players=3)

custom_rules = {
    'main_instrument_priority': True,        # Prioritize main instruments
    'accessory_flexibility': True,          # Allow accessory sharing
    'simultaneous_conflict_penalty': 5.0,   # Penalty for timing conflicts
    'player_balance_weight': 1.0,           # Workload balance importance
    'complexity_balance_weight': 1.5        # Complexity balance importance
}

distributor.set_distribution_rules(custom_rules)
```

### Instrument Characteristics

Each instrument is classified based on:

- **Pitch Range** - MIDI note numbers
- **Staff Position** - Location on percussion staff
- **Note Head Type** - Normal, X, diamond shapes
- **Typical Dynamics** - Expected volume levels

## Performance Considerations

- **Timing Conflicts**: System minimizes simultaneous notes for same player
- **Physical Limitations**: Considers hand-crossing and instrument positioning
- **Workload Balance**: Distributes notes evenly when possible
- **Musical Practicality**: Maintains musical coherence and playability

## Examples

See `examples/percussion_usage.py` for comprehensive examples including:

- Basic percussion distribution
- Main vs accessory instrument rules
- Complex multi-instrument scores
- Custom distribution rules
- Analysis result interpretation

## Troubleshooting

### Common Issues

1. **Too Many Main Instruments**: If more main instruments than players, system creates overflow assignments
2. **Timing Conflicts**: Adjust `simultaneous_conflict_penalty` to reduce conflicts
3. **Unbalanced Distribution**: Increase `player_balance_weight` for more even distribution

### Debug Information

Enable detailed logging:

```bash
export LOG_LEVEL=debug
python -m oemer.enhanced_ete score.jpg --num-players 3
```

## Contributing

Areas for improvement:
- Additional percussion instruments
- Better instrument classification
- Performance optimization
- Extended accessory instrument support

## License

MIT License (same as original Oemer project)