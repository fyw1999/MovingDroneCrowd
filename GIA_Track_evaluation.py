import os
import sys
import shutil
import argparse
import pandas as pd
import glob

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate MovingDroneCrowd++ using TrackEval (Command Line Version)')
    
    # ================= Required path arguments =================
    parser.add_argument('--trackeval_path', type=str, default="./TrackEval",
                        help='Root directory of the TrackEval library')
    
    parser.add_argument('--dataset_root', type=str, default="",
                        help='')
    
    parser.add_argument('--pred_root', type=str, default="",
                        help='')
    
    parser.add_argument('--split_file', type=str, default="test.txt",
                        help='')
    
    # ================= Required hyperparameters =================
    parser.add_argument('--interval', type=int, default=4,
                        help='Frame interval of predictions (GT is sampled at this interval)')
    
    parser.add_argument('--fixed_w', type=float, default=20.0,
                        help='Fixed width override (used to compute IoU)')
    
    parser.add_argument('--fixed_h', type=float, default=20.0,
                        help='Fixed height override (used to compute IoU)')
    
    # ================= Optional arguments =================
    parser.add_argument('--temp_dir', type=str, default="./temp_eval_workspace",
                        help='Directory for generated temporary evaluation files')
    
    return parser.parse_args()

# Parse arguments
args = parse_args()

# Add the TrackEval path dynamically
sys.path.append(os.path.abspath(args.trackeval_path))

try:
    from TrackEval.trackeval.eval import Evaluator
    from TrackEval.trackeval.datasets.mot_challenge_2d_box import MotChallenge2DBox
    from TrackEval.trackeval.metrics import HOTA, CLEAR, Identity
except ImportError:
    print(f"\n❌ Error: Failed to import TrackEval.")
    print(f"Please check whether the path is correct: {os.path.abspath(args.trackeval_path)}")
    print("Alternatively, run: git clone https://github.com/JonathonLuiten/TrackEval.git")
    sys.exit(1)

def get_sequences_from_split(dataset_root, split_file):
    split_path = os.path.join(dataset_root, split_file)
    if not os.path.exists(split_path):
        print(f"❌ Error: Split file not found: {split_path}")
        sys.exit(1)

    sequences = []
    with open(split_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    for line in lines:
        if '/' in line:
            # Format: scene_1/001
            parts = line.split('/')
            sequences.append((parts[0], parts[1]))
        else:
            # Format: scene_1 (read all clips in this scene)
            scene_name = line
            # Support both annotation and annotations directory conventions
            search_path = os.path.join(dataset_root, 'annotations', scene_name)
            
            if os.path.isdir(search_path):
                files = [f for f in os.listdir(search_path) if f.endswith('.csv')]
                for fname in sorted(files):
                    clip_name = os.path.splitext(fname)[0]
                    sequences.append((scene_name, clip_name))
    return sequences

def convert_file(src_path, dst_path, interval=1, is_gt=False, fixed_w=20, fixed_h=20):
    """
    Read CSV/TXT data, handle frame intervals, and recompute bounding boxes from center points.
    """
    try:
        # 1. Detect the delimiter automatically
        delimiter = None
        with open(src_path, 'r') as f:
            line = f.readline()
            delimiter = ',' if ',' in line else None
        
        # 2. Read the data
        try:
            df = pd.read_csv(src_path, header=None, sep=delimiter, engine='python')
        except pd.errors.EmptyDataError:
            with open(dst_path, 'w') as f: pass
            return True, 0

        df = df.dropna()
        
        # 3. Filter GT frames (temporal downsampling)
        if is_gt:
            # Original GT: 0, 1, 2, 3, 4, ...
            # Retained frames: 0, 4, 8, ...
            df = df[df[0] % interval == 0].copy()
            # Align frame indices: 0->1, 4->5
            df[0] = df[0] + 1
        
        # 4. Remove duplicates (Frame and ID must be unique)
        if df.duplicated(subset=[0, 1]).any():
            df = df.drop_duplicates(subset=[0, 1], keep='first')

        max_frame = 0
        
        # 5. Convert and write the data
        with open(dst_path, 'w') as f:
            for _, row in df.iterrows():
                frame = int(float(row[0]))
                if frame > max_frame: max_frame = frame
                pid = int(float(row[1]))
                
                # Read the original coordinates
                val_x, val_y = float(row[2]), float(row[3])
                org_w = float(row[4]) if len(row) > 4 else 0
                org_h = float(row[5]) if len(row) > 5 else 0
                
                # --- Core logic: recompute the box from its center point ---
                # Assume val_x and val_y represent the top-left corner
                cx = val_x + org_w / 2.0
                cy = val_y + org_h / 2.0
                
                # Compute the new top-left corner from the fixed width and height
                new_left = cx - fixed_w / 2.0
                new_top = cy - fixed_h / 2.0
                
                # Write MOT format: frame, id, x, y, w, h, conf, -1, -1, -1
                f.write(f"{frame},{pid},{new_left:.2f},{new_top:.2f},{fixed_w},{fixed_h},1,-1,-1,-1\n")
                
        return True, max_frame
    except Exception as e:
        print(f"⚠️ Failed to convert {src_path}: {e}")
        with open(dst_path, 'w') as f: pass
        return False, 0

def prepare_data(sequences, args):
    """Build the TrackEval directory structure."""
    temp_dir = os.path.abspath(args.temp_dir)
    print(f"🔨 Building the evaluation environment at: {temp_dir}")
    
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    
    # Define the benchmark and split names
    BENCHMARK = "MyBench"
    SPLIT = "test" # Keep this as "test" to match the TrackEval configuration
    COMBINED_NAME = f"{BENCHMARK}-{SPLIT}" 
    
    # Directory structure
    gt_base_dir = os.path.join(temp_dir, "data", "gt", "mot_challenge", COMBINED_NAME)
    tracker_name = "MyTracker"
    pred_base_dir = os.path.join(temp_dir, "data", "trackers", "mot_challenge", COMBINED_NAME, tracker_name, "data")
    
    os.makedirs(pred_base_dir, exist_ok=True)
    
    valid_seq_names = []

    for scene, clip in sequences:
        seq_name = f"{scene}_{clip}"
        
        gt_src = os.path.join(args.dataset_root, 'annotations', scene, f"{clip}.csv")
        pred_src = os.path.join(args.pred_root, scene, f"{clip}.txt")
        
        if not os.path.exists(gt_src):
            continue
            
        # 1. Convert GT
        gt_dst_dir = os.path.join(gt_base_dir, seq_name, "gt")
        os.makedirs(gt_dst_dir, exist_ok=True)
        
        success, max_frame = convert_file(
            gt_src, 
            os.path.join(gt_dst_dir, "gt.txt"), 
            interval=args.interval, 
            is_gt=True, 
            fixed_w=args.fixed_w, 
            fixed_h=args.fixed_h
        )
        
        if not success: continue

        # 2. Generate seqinfo.ini
        with open(os.path.join(gt_base_dir, seq_name, "seqinfo.ini"), 'w') as f:
            f.write(f"[Sequence]\nname={seq_name}\nimDir=img1\nframeDir=img1\n")
            f.write(f"seqLength={max_frame+100}\nimWidth=1920\nimHeight=1080\nimExt=.jpg\n")

        # 3. Convert predictions
        pred_dst = os.path.join(pred_base_dir, f"{seq_name}.txt")
        if os.path.exists(pred_src):
            convert_file(
                pred_src, 
                pred_dst, 
                interval=1, # Predictions are already sparse, so no filtering is needed
                is_gt=False, 
                fixed_w=args.fixed_w, 
                fixed_h=args.fixed_h
            )
        else:
            open(pred_dst, 'w').close()
            
        valid_seq_names.append(seq_name)

    # 4. Generate the seqmap file
    seqmap_dir = os.path.join(temp_dir, "data", "gt", "mot_challenge", "seqmaps")
    os.makedirs(seqmap_dir, exist_ok=True)
    seqmap_file = os.path.join(seqmap_dir, f"{COMBINED_NAME}.txt")
    
    with open(seqmap_file, 'w') as f:
        f.write("name\n")
        for seq in valid_seq_names:
            f.write(f"{seq}\n")
            
    print(f"📄 Generated seqmap: {seqmap_file} (contains {len(valid_seq_names)} sequences)")
    return valid_seq_names

def run_eval(args):
    temp_dir = os.path.abspath(args.temp_dir)
    
    eval_config = eval_config = {
        'USE_PARALLEL': False,
        'NUM_PARALLEL_CORES': 1,
        'PRINT_RESULTS': True,
        'PRINT_ONLY_COMBINED': False,
        'OUTPUT_SUMMARY': True,
        'PLOT_CURVES': False,
    }
    eval_config['PRINT_CONFIG'] = False
    
    dataset_config = {
        'GT_FOLDER': os.path.join(args.temp_dir, 'data', 'gt', 'mot_challenge'),
        'TRACKERS_FOLDER': os.path.join(args.temp_dir, 'data', 'trackers', 'mot_challenge'),
        'OUTPUT_FOLDER': None,
        'TRACKERS_TO_EVAL': ['MyTracker'],
        'CLASSES_TO_EVAL': ['pedestrian'],
        'BENCHMARK': 'MyBench',  # Prefix of the MyBench-test.txt filename
        'SPLIT_TO_EVAL': 'test', # Suffix of the MyBench-test.txt filename
        'INPUT_AS_ZIP': False,
        'DO_PREPROC': False,
        'TRACKER_SUB_FOLDER': 'data',
        'OUTPUT_SUB_FOLDER': '',
        # 'SEQMAP_FILE_LIST': ... # Not needed: use either the default file or the file generated above
    }
    
    dataset_config['SKIP_SPLIT_FOL'] = False 
    
    evaluator = Evaluator(eval_config)
    dataset_list = [MotChallenge2DBox(dataset_config)]
    metrics_list = [HOTA(), CLEAR(), Identity()]
    
    print("\n" + "="*50)
    print(f"🚀 Starting evaluation | Interval: {args.interval} | BoxSize: {args.fixed_w}x{args.fixed_h}")
    print("="*50 + "\n")
    
    evaluator.evaluate(dataset_list, metrics_list)

if __name__ == "__main__":
    print(f"📂 Dataset path: {args.dataset_root}")
    print(f"📂 Prediction path: {args.pred_root}")
    
    target_sequences = get_sequences_from_split(args.dataset_root, args.split_file)
    
    if not target_sequences:
        print("❌ No sequences found. Please check the split file.")
        sys.exit(0)
        
    print(f"🔍 Found {len(target_sequences)} sequences to evaluate.")
    
    valid_seqs = prepare_data(target_sequences, args)
    
    if valid_seqs:
        run_eval(args)
        print(f"\n✅ Evaluation completed!")
    else:
        print("❌ Unable to generate valid evaluation data.")
