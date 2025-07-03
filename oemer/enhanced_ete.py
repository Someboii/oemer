import os
import pickle
import argparse
from pathlib import Path
from typing import Dict, Any

from PIL import Image
import cv2
import numpy as np

from oemer import MODULE_PATH, layers
from oemer.inference import inference
from oemer.logger import get_logger
from oemer.dewarp import estimate_coords, dewarp
from oemer.staffline_extraction import extract as staff_extract
from oemer.notehead_extraction import extract as note_extract
from oemer.note_group_extraction import extract as group_extract
from oemer.symbol_extraction import extract as symbol_extract
from oemer.rhythm_extraction import extract as rhythm_extract
from oemer.build_system import MusicXMLBuilder
from oemer.instrument_recognition import analyze_score_with_ml
from oemer.draw_teaser import teaser

logger = get_logger(__name__)


def enhanced_extract(args) -> Dict[str, Any]:
    """Enhanced extraction with ML analysis"""
    
    # Original OMR processing
    img_path = Path(args.img_path)
    f_name = os.path.splitext(img_path.name)[0]
    pkl_path = img_path.parent / f"{f_name}.pkl"
    
    if pkl_path.exists():
        # Load from cache
        pred = pickle.load(open(pkl_path, "rb"))
        notehead = pred["note"]
        symbols = pred["symbols"]
        staff = pred["staff"]
        clefs_keys = pred["clefs_keys"]
        stems_rests = pred["stems_rests"]
    else:
        # Make predictions
        if args.use_tf:
            ori_inf_type = os.environ.get("INFERENCE_WITH_TF", None)
            os.environ["INFERENCE_WITH_TF"] = "true"
        
        staff, symbols, stems_rests, notehead, clefs_keys = generate_pred(str(img_path), use_tf=args.use_tf)
        
        if args.use_tf and ori_inf_type is not None:
            os.environ["INFERENCE_WITH_TF"] = ori_inf_type
        
        if args.save_cache:
            data = {
                'staff': staff,
                'note': notehead,
                'symbols': symbols,
                'stems_rests': stems_rests,
                'clefs_keys': clefs_keys
            }
            pickle.dump(data, open(pkl_path, "wb"))

    # Load and process image
    image_pil = Image.open(str(img_path))
    if "GIF" != image_pil.format:
        image = cv2.imread(str(img_path))
    else:
        gif_image = image_pil.convert('RGB')
        gif_img_arr = np.array(gif_image)
        image = gif_img_arr[:, :, ::-1].copy()

    image = cv2.resize(image, (staff.shape[1], staff.shape[0]))

    # Dewarping
    if not args.without_deskew:
        logger.info("Dewarping")
        coords_x, coords_y = estimate_coords(staff)
        staff = dewarp(staff, coords_x, coords_y)
        symbols = dewarp(symbols, coords_x, coords_y)
        stems_rests = dewarp(stems_rests, coords_x, coords_y)
        clefs_keys = dewarp(clefs_keys, coords_x, coords_y)
        notehead = dewarp(notehead, coords_x, coords_y)
        for i in range(image.shape[2]):
            image[..., i] = dewarp(image[..., i], coords_x, coords_y)

    # Register predictions
    symbols = symbols + clefs_keys + stems_rests
    symbols[symbols>1] = 1
    layers.register_layer("stems_rests_pred", stems_rests)
    layers.register_layer("clefs_keys_pred", clefs_keys)
    layers.register_layer("notehead_pred", notehead)
    layers.register_layer("symbols_pred", symbols)
    layers.register_layer("staff_pred", staff)
    layers.register_layer("original_image", image)

    # Extract musical elements
    logger.info("Extracting stafflines")
    staffs, zones = staff_extract()
    layers.register_layer("staffs", staffs)
    layers.register_layer("zones", zones)

    logger.info("Extracting noteheads")
    notes = note_extract()
    layers.register_layer('notes', np.array(notes))

    # Register note IDs
    layers.register_layer('note_id', np.zeros(symbols.shape, dtype=np.int64)-1)
    register_note_id()

    logger.info("Grouping noteheads")
    groups, group_map = group_extract()
    layers.register_layer('note_groups', np.array(groups))
    layers.register_layer('group_map', group_map)

    logger.info("Extracting symbols")
    barlines, clefs, sfns, rests = symbol_extract()
    layers.register_layer('barlines', np.array(barlines))
    layers.register_layer('clefs', np.array(clefs))
    layers.register_layer('sfns', np.array(sfns))
    layers.register_layer('rests', np.array(rests))

    logger.info("Extracting rhythm types")
    rhythm_extract()

    # Prepare score data for ML analysis
    score_data = {
        'notes': notes,
        'note_groups': groups,
        'staffs': staffs,
        'clefs': clefs,
        'sfns': sfns,
        'rests': rests,
        'barlines': barlines,
        'measures': {}  # This would be populated with measure information
    }

    # ML Analysis
    logger.info("Performing ML analysis for instrument recognition and voice distribution")
    ml_analysis = analyze_score_with_ml(score_data, num_players=args.num_players)

    # Build MusicXML with enhanced information
    logger.info("Building enhanced MusicXML document")
    basename = os.path.basename(img_path).replace(".jpg", "").replace(".png", "")
    builder = MusicXMLBuilder(title=basename.capitalize())
    builder.build()
    xml = builder.to_musicxml()

    # Write output files
    out_path = args.output_path
    if not out_path.endswith(".musicxml"):
        out_path = os.path.join(out_path, basename+".musicxml")

    with open(out_path, "wb") as ff:
        ff.write(xml)

    # Save ML analysis results
    analysis_path = out_path.replace(".musicxml", "_analysis.json")
    save_analysis_results(ml_analysis, analysis_path)

    # Generate visualization
    img = teaser()
    img.save(out_path.replace(".musicxml", "_analysis.png"))

    return {
        'musicxml_path': out_path,
        'analysis_path': analysis_path,
        'ml_analysis': ml_analysis
    }


def generate_pred(img_path: str, use_tf: bool = False):
    """Generate predictions using the neural networks"""
    logger.info("Extracting staffline and symbols")
    staff_symbols_map, _ = inference(
        os.path.join(MODULE_PATH, "checkpoints/unet_big"),
        img_path,
        use_tf=use_tf,
    )
    staff = np.where(staff_symbols_map==1, 1, 0)
    symbols = np.where(staff_symbols_map==2, 1, 0)

    logger.info("Extracting layers of different symbols")
    sep, _ = inference(
        os.path.join(MODULE_PATH, "checkpoints/seg_net"),
        img_path,
        manual_th=None,
        use_tf=use_tf,
    )
    stems_rests = np.where(sep==1, 1, 0)
    notehead = np.where(sep==2, 1, 0)
    clefs_keys = np.where(sep==3, 1, 0)

    return staff, symbols, stems_rests, notehead, clefs_keys


def register_note_id():
    """Register note IDs in the layer"""
    symbols = layers.get_layer('symbols_pred')
    layer = layers.get_layer('note_id')
    notes = layers.get_layer('notes')
    
    for idx, note in enumerate(notes):
        x1, y1, x2, y2 = note.bbox
        yi, xi = np.where(symbols[y1:y2, x1:x2]>0)
        yi += y1
        xi += x1
        layer[yi, xi] = idx
        notes[idx].id = idx


def save_analysis_results(analysis: Dict[str, Any], filepath: str):
    """Save ML analysis results to JSON file"""
    import json
    
    # Convert complex objects to serializable format
    serializable_analysis = {
        'instrument': analysis['instrument'].value,
        'total_notes': len(analysis['note_events']),
        'simultaneous_events': len(analysis['simultaneous_notes']),
        'voice_count': len(analysis['voices']),
        'voice_statistics': analysis['voice_statistics'],
        'bar_complexities': analysis['bar_complexities'],
        'voice_details': []
    }
    
    # Add voice details
    for voice in analysis['voices']:
        voice_detail = {
            'voice_id': voice.voice_id,
            'player_id': voice.player_id,
            'note_count': len(voice.note_events),
            'complexity_score': voice.complexity_score,
            'note_events': [
                {
                    'bar': event.bar_number,
                    'beat': event.beat_position,
                    'pitch': event.pitch,
                    'duration': event.duration,
                    'velocity': event.velocity
                }
                for event in voice.note_events[:10]  # Limit to first 10 for brevity
            ]
        }
        serializable_analysis['voice_details'].append(voice_detail)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_analysis, f, indent=2)
    
    logger.info(f"Analysis results saved to {filepath}")


def get_enhanced_parser():
    """Get argument parser with enhanced options"""
    parser = argparse.ArgumentParser(
        "Enhanced Oemer",
        description="Enhanced OMR with ML instrument recognition and voice distribution",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("img_path", help="Path to the image.", type=str)
    parser.add_argument(
        "-o", "--output-path", help="Path to output the result file.", type=str, default="./")
    parser.add_argument(
        "--use-tf", help="Use Tensorflow for model inference.", action="store_true")
    parser.add_argument(
        "--save-cache", help="Save model predictions for reuse.", action='store_true')
    parser.add_argument(
        "-d", "--without-deskew", help="Disable deskewing step.", action='store_true')
    parser.add_argument(
        "--num-players", help="Number of players available for percussion.", type=int, default=2)
    parser.add_argument(
        "--distribution-rules", help="JSON file with custom distribution rules.", type=str, default=None)
    
    return parser


def main():
    """Enhanced main function"""
    parser = get_enhanced_parser()
    args = parser.parse_args()

    if not os.path.exists(args.img_path):
        raise FileNotFoundError(f"Image path doesn't exist: {args.img_path}")

    # Clear previous data
    clear_data()
    
    # Process with enhanced features
    result = enhanced_extract(args)
    
    logger.info(f"Enhanced OMR processing complete!")
    logger.info(f"MusicXML saved to: {result['musicxml_path']}")
    logger.info(f"Analysis saved to: {result['analysis_path']}")
    logger.info(f"Detected instrument: {result['ml_analysis']['instrument'].value}")
    logger.info(f"Generated {len(result['ml_analysis']['voices'])} voices for {args.num_players} players")


def clear_data():
    """Clear layer data"""
    lls = layers.list_layers()
    for l in lls:
        layers.delete_layer(l)


if __name__ == "__main__":
    main()