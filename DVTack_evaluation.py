import os
import sys
import shutil
import argparse
import pandas as pd
import glob

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate MovingDroneCrowd++ using TrackEval (Command Line Version)')
    
    # ================= 必需路径参数 =================
    parser.add_argument('--trackeval_path', type=str, default="./TrackEval",
                        help='TrackEval 库的根目录路径')
    
    parser.add_argument('--dataset_root', type=str, default="",
                        help='')
    
    parser.add_argument('--pred_root', type=str, default="",
                        help='')
    
    parser.add_argument('--split_file', type=str, default="test.txt",
                        help='')
    
    # ================= 必需超参数 =================
    parser.add_argument('--interval', type=int, default=4,
                        help='预测结果的帧间隔 (GT会根据此间隔进行采样)')
    
    parser.add_argument('--fixed_w', type=float, default=20.0,
                        help='强制设置的固定宽度 (用于计算 IoU)')
    
    parser.add_argument('--fixed_h', type=float, default=20.0,
                        help='强制设置的固定高度 (用于计算 IoU)')
    
    # ================= 可选参数 =================
    parser.add_argument('--temp_dir', type=str, default="./temp_eval_workspace",
                        help='生成的临时评估文件存放目录')
    
    return parser.parse_args()

# 解析参数
args = parse_args()

# 动态添加 TrackEval 路径
sys.path.append(os.path.abspath(args.trackeval_path))

try:
    from TrackEval.trackeval.eval import Evaluator
    from TrackEval.trackeval.datasets.mot_challenge_2d_box import MotChallenge2DBox
    from TrackEval.trackeval.metrics import HOTA, CLEAR, Identity
except ImportError:
    print(f"\n❌ 错误: 无法导入 TrackEval。")
    print(f"请检查路径是否正确: {os.path.abspath(args.trackeval_path)}")
    print("或者运行: git clone https://github.com/JonathonLuiten/TrackEval.git")
    sys.exit(1)

def get_sequences_from_split(dataset_root, split_file):
    split_path = os.path.join(dataset_root, split_file)
    if not os.path.exists(split_path):
        print(f"❌ 错误: 找不到划分文件 {split_path}")
        sys.exit(1)

    sequences = []
    with open(split_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    for line in lines:
        if '/' in line:
            # 格式: scene_1/001
            parts = line.split('/')
            sequences.append((parts[0], parts[1]))
        else:
            # 格式: scene_1 (读取该场景下所有 clip)
            scene_name = line
            # 兼容 annotation 和 annotations
            search_path = os.path.join(dataset_root, 'annotations', scene_name)
            
            if os.path.isdir(search_path):
                files = [f for f in os.listdir(search_path) if f.endswith('.csv')]
                for fname in sorted(files):
                    clip_name = os.path.splitext(fname)[0]
                    sequences.append((scene_name, clip_name))
    return sequences

def convert_file(src_path, dst_path, interval=1, is_gt=False, fixed_w=20, fixed_h=20):
    """
    读取 CSV/TXT，处理帧间隔，并基于中心点重算 BBox
    """
    try:
        # 1. 自动探测分隔符
        delimiter = None
        with open(src_path, 'r') as f:
            line = f.readline()
            delimiter = ',' if ',' in line else None
        
        # 2. 读取数据
        try:
            df = pd.read_csv(src_path, header=None, sep=delimiter, engine='python')
        except pd.errors.EmptyDataError:
            with open(dst_path, 'w') as f: pass
            return True, 0

        df = df.dropna()
        
        # 3. 筛选 GT 帧 (稀疏化)
        if is_gt:
            # GT 原始: 0, 1, 2, 3, 4 ...
            # 筛选: 0, 4, 8 ...
            df = df[df[0] % interval == 0].copy()
            # 帧号对齐: 0->1, 4->5
            df[0] = df[0] + 1
        
        # 4. 去重 (Frame, ID 唯一)
        if df.duplicated(subset=[0, 1]).any():
            df = df.drop_duplicates(subset=[0, 1], keep='first')

        max_frame = 0
        
        # 5. 转换并写入
        with open(dst_path, 'w') as f:
            for _, row in df.iterrows():
                frame = int(float(row[0]))
                if frame > max_frame: max_frame = frame
                pid = int(float(row[1]))
                
                # 读取原始坐标
                val_x, val_y = float(row[2]), float(row[3])
                org_w = float(row[4]) if len(row) > 4 else 0
                org_h = float(row[5]) if len(row) > 5 else 0
                
                # --- 核心逻辑: 基于中心点重算 ---
                # 假设 val_x, val_y 是左上角
                cx = val_x + org_w / 2.0
                cy = val_y + org_h / 2.0
                
                # 根据新的固定宽高，反算新的左上角
                new_left = cx - fixed_w / 2.0
                new_top = cy - fixed_h / 2.0
                
                # 写入 MOT 格式: frame, id, x, y, w, h, conf, -1, -1, -1
                f.write(f"{frame},{pid},{new_left:.2f},{new_top:.2f},{fixed_w},{fixed_h},1,-1,-1,-1\n")
                
        return True, max_frame
    except Exception as e:
        print(f"⚠️ 转换失败 {src_path}: {e}")
        with open(dst_path, 'w') as f: pass
        return False, 0

def prepare_data(sequences, args):
    """构建 TrackEval 目录结构"""
    temp_dir = os.path.abspath(args.temp_dir)
    print(f"🔨 正在构建评估环境到: {temp_dir}")
    
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    
    # 定义 Benchmark 和 Split 名称
    BENCHMARK = "MyBench"
    SPLIT = "test" # 这里固定叫 test，对应 TrackEval config
    COMBINED_NAME = f"{BENCHMARK}-{SPLIT}" 
    
    # 目录结构
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
            
        # 1. 转换 GT
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

        # 2. 生成 seqinfo.ini
        with open(os.path.join(gt_base_dir, seq_name, "seqinfo.ini"), 'w') as f:
            f.write(f"[Sequence]\nname={seq_name}\nimDir=img1\nframeDir=img1\n")
            f.write(f"seqLength={max_frame+100}\nimWidth=1920\nimHeight=1080\nimExt=.jpg\n")

        # 3. 转换 Pred
        pred_dst = os.path.join(pred_base_dir, f"{seq_name}.txt")
        if os.path.exists(pred_src):
            convert_file(
                pred_src, 
                pred_dst, 
                interval=1, # Pred 已经稀疏，不需要筛选
                is_gt=False, 
                fixed_w=args.fixed_w, 
                fixed_h=args.fixed_h
            )
        else:
            open(pred_dst, 'w').close()
            
        valid_seq_names.append(seq_name)

    # 4. 生成 seqmap 文件
    seqmap_dir = os.path.join(temp_dir, "data", "gt", "mot_challenge", "seqmaps")
    os.makedirs(seqmap_dir, exist_ok=True)
    seqmap_file = os.path.join(seqmap_dir, f"{COMBINED_NAME}.txt")
    
    with open(seqmap_file, 'w') as f:
        f.write("name\n")
        for seq in valid_seq_names:
            f.write(f"{seq}\n")
            
    print(f"📄 生成 Seqmap: {seqmap_file} (包含 {len(valid_seq_names)} 个序列)")
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
        'BENCHMARK': 'MyBench',  # 对应文件名 MyBench-test.txt 的前缀
        'SPLIT_TO_EVAL': 'test', # 对应文件名 MyBench-test.txt 的后缀
        'INPUT_AS_ZIP': False,
        'DO_PREPROC': False,
        'TRACKER_SUB_FOLDER': 'data',
        'OUTPUT_SUB_FOLDER': '',
        # 'SEQMAP_FILE_LIST': ... # 不需要传这个了，因为我们要么用默认文件，要么上面已经生成了文件
    }
    
    dataset_config['SKIP_SPLIT_FOL'] = False 
    
    evaluator = Evaluator(eval_config)
    dataset_list = [MotChallenge2DBox(dataset_config)]
    metrics_list = [HOTA(), CLEAR(), Identity()]
    
    print("\n" + "="*50)
    print(f"🚀 开始评估 | Interval: {args.interval} | BoxSize: {args.fixed_w}x{args.fixed_h}")
    print("="*50 + "\n")
    
    evaluator.evaluate(dataset_list, metrics_list)

if __name__ == "__main__":
    print(f"📂 数据集路径: {args.dataset_root}")
    print(f"📂 预测结果路径: {args.pred_root}")
    
    target_sequences = get_sequences_from_split(args.dataset_root, args.split_file)
    
    if not target_sequences:
        print("❌ 未找到任何序列，请检查 split 文件。")
        sys.exit(0)
        
    print(f"🔍 发现 {len(target_sequences)} 个待评估序列。")
    
    valid_seqs = prepare_data(target_sequences, args)
    
    if valid_seqs:
        run_eval(args)
        print(f"\n✅ 评估完成！")
    else:
        print("❌ 无法生成有效评估数据。")