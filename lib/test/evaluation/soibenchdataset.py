import os
import json
import numpy as np
import pandas as pd  # VideoCube 依赖
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
# 1. 更新 Imports：添加了 load_str
from lib.test.utils.load_text import load_text, load_str
import pprint # 如果文件头部没导入，可以在这里导入，或者放到文件最上面


################################################################################
# 
# 章节 1: 新基准类 (SOIBenchmark)
#
################################################################################

class SOIBenchDataset(BaseDataset):
    """
    SOIBenchmark 数据集加载器 (修复版)
    适配文件名格式: {sequence_name}_descriptions.jsonl
    """
    
    def __init__(self):
        super().__init__()

        # SOIBench 标注根目录（在 lib/test/evaluation/local.py 里统一维护）
        root_path = getattr(
            self.env_settings,
            'soi_bench_path',
            '/root/user-data/PUBLIC_DATASETS/SOIBench/test_anno'
        )
        
        self.dataset_loaders = {
            'lasot': {
                'loader': LaSOTDataset(), 
                'soi_dir': os.path.join(root_path, "lasot")
            },
            'videocube': {
                'loader': VideoCubeDataset(split='test', version='tiny'), 
                'soi_dir': os.path.join(root_path, "mgit")
            },
            'tnl2k': {
                'loader': TNL2kDataset(),
                'soi_dir': os.path.join(root_path, "tnl2k")
            },
        }

        self.soi_settings = {
            #
            "use_original_text": False,  # True=强制使用原数据集文本(消融用); False=使用SOI标注

            # 决定使用哪些层级
            "levels": [1, 2, 3, 4], 
            
            # 模式选择: 'realtime' | 'fixed' | 'hierarchical'
            "persistence_mode": "realtime",  # "hierarchical", 
            
            # 仅当 mode='fixed' 时生效
            "fixed_duration": 30, 

            # 仅当 mode='hierarchical' 时生效
            # 这里对应您的假设：L1/L2(外观)更持久，L3(动作)次之，L4(环境)最短
            "level_durations": {
                1: 999999,  # 近似永久 (直到被新标注覆盖)
                2: 999999,
                3: 60,      # 动作持续约 2秒
                4: 30       # 环境/方位持续约 1秒
            }
        }

        # ================= [新增] 打印 SOI 配置信息 =================
        self._print_config_info()
        # ==========================================================
        
        self.sequence_list = self._get_sequence_list()
    
    def _print_config_info(self):
        """打印SOI配置信息,确保消融实验配置清晰可见"""
        print("\n" + "="*80)
        print(" " * 25 + "🔍 SOI BENCHMARK CONFIGURATION 🔍")
        print("="*80)
        
        # 生成并打印配置摘要
        summary = self._generate_config_summary()
        print(f"\n📋 Configuration Summary:\n   {summary}\n")
        
        # 打印详细配置
        print("📝 Detailed Settings:")
        print("-" * 80)
        for key, value in self.soi_settings.items():
            if key == 'level_durations' and isinstance(value, dict):
                print(f"  {key}:")
                for level, duration in sorted(value.items()):
                    print(f"    Level {level}: {duration} frames")
            else:
                print(f"  {key}: {value}")
        print("="*80)
        
    
    def _generate_config_summary(self):
        """生成简洁的配置摘要"""
        if self.soi_settings.get('use_original_text', False):
            return "⚠️  ABLATION - Original Text Only (SOI Disabled)"
        
        mode = self.soi_settings.get('persistence_mode', 'fixed')
        levels = self.soi_settings.get('levels', [1, 2, 3, 4])
        
        if mode == 'realtime':
            duration_info = "Realtime 只在当前SOI帧有效，否则回退原文本"
        elif mode == 'fixed':
            fixed_dur = self.soi_settings.get('fixed_duration', 30)
            duration_info = f"Fixed duration ({fixed_dur} frames for all levels) SOI帧文本持续n帧"
        elif mode == 'hierarchical':
            level_durs = self.soi_settings.get('level_durations', {})
            dur_parts = [f"L{l}:{level_durs.get(l, 30)}f" for l in levels]
            duration_info = f"Hierarchical ({', '.join(dur_parts)}) SOI帧文本根据层级进行持续作用"
        else:
            duration_info = f"Unknown mode: {mode}"
        
        return f"Levels: {levels} | Mode: {duration_info}"

    def get_sequence_list(self):
        return SequenceList(self.sequence_list)

    def _get_sequence_list(self):
        all_sequences = []
        print(f"Loading SOIBenchmark (JSONL version)...")

        for dset_name, config in self.dataset_loaders.items():
            loader_instance = config['loader']
            soi_dir_path = config['soi_dir']
            
            if not os.path.exists(soi_dir_path):
                print(f"  ! Warning: Directory not found: {soi_dir_path}")
                continue

            print(f"  > Scanning {dset_name} from {soi_dir_path} ...")
            
            # --- 修改点 1: 只筛选以 _descriptions.jsonl 结尾的文件 ---
            file_list = [f for f in os.listdir(soi_dir_path) if f.endswith('_descriptions.jsonl')]
            
            valid_count = 0
            for filename in sorted(file_list):
                # --- 修改点 2: 正确解析序列名 ---
                # 例如: "airplane-1_descriptions.jsonl" -> "airplane-1"
                seq_name = filename.replace('_descriptions.jsonl', '').replace('frame_', '')
                
                # try:
                # 构造序列 (加载 GT, 图片路径等)
                sequence = loader_instance._construct_sequence(seq_name)
                
                # 读取并注入 SOI 数据
                full_jsonl_path = os.path.join(soi_dir_path, filename)
                processed_soi_data = self._read_and_process_soi(full_jsonl_path, sequence)
                sequence.soi_info = processed_soi_data
                # soi_data = self._read_jsonl(full_jsonl_path)
                
                sequence.dataset = 'soibench'
                sequence.original_dataset = dset_name

                all_sequences.append(sequence)
                valid_count += 1
                    
                # except Exception as e:
                #     # 如果 construct_sequence 失败（例如原数据集路径里没有这个序列），会在这里报错
                #     print(f"  ! Error loading sequence '{seq_name}' from {dset_name}: {e}")

            print(f"  > Loaded {valid_count} sequences from {dset_name}.")

        print(f"SOIBenchmark loaded. Total sequences: {len(all_sequences)}")
        return all_sequences

    def _read_jsonl(self, file_path):
        """
        读取 jsonl 文件，返回列表
        """
        data_list = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    data_list.append(data)
                except json.JSONDecodeError as e:
                    print(f"    ! JSON Parse Error in {file_path}: {e}")
        return data_list

    def __len__(self):
        return len(self.sequence_list)
    
    def _read_and_process_soi(self, file_path, sequence):
        """
        读取 SOI 标注 (JSONL格式)，支持：
        1. 分层级文本拼接 (Levels)
        2. 分层级时效性控制 (Hierarchical Persistence)
        3. 缺失值回退策略 (Fallback to Original Text)
        """

        # ==========================
        # 1. 提取原始文本 (Fallback / Original)
        # ==========================
        # 不管开不开消融，先把原始文本拿出来
        original_text = ""
        if hasattr(sequence, 'language_query') and sequence.language_query:
            original_text = str(sequence.language_query)
        elif hasattr(sequence, 'object_class') and sequence.object_class:
            original_text = str(sequence.object_class)
        
        if original_text:
            original_text = original_text.strip()
            if original_text:
                original_text = original_text[0].upper() + original_text[1:]
        
        total_frames = len(sequence.frames)

        # ==========================
        # [新增] 原始文本消融逻辑
        # ==========================
        # 如果开启了 "使用原始文本"，则直接忽略 SOI JSONL 文件
        # 每一帧都填入原始文本，且 need_update 只有第0帧为 True
        if self.soi_settings.get('use_original_text', False):
            
            soi_list = []
            for i in range(total_frames):
                # 只有第0帧需要更新特征，后面全是静态文本，不需要更新
                need_upd = (i == 0) 
                soi_list.append({
                    'text': original_text,
                    'need_update': need_upd
                })
            return soi_list

        # ==========================
        # 2. 准备配置与 Fallback 文本
        # ==========================
        req_levels = self.soi_settings.get('levels', [1, 2, 3, 4])
        lang = self.soi_settings.get('language', 'en')
        lang_root_key = f"output-{lang}"
        mode = self.soi_settings.get('persistence_mode', 'fixed')
        
        # 获取原始数据集的文本作为“保底”
        fallback_text = ""
        if hasattr(sequence, 'language_query') and sequence.language_query:
            fallback_text = str(sequence.language_query)
        elif hasattr(sequence, 'object_class') and sequence.object_class:
            fallback_text = str(sequence.object_class)
        
        if fallback_text:
            fallback_text = fallback_text.strip()
            fallback_text = fallback_text[0].upper() + fallback_text[1:]

        # ==========================
        # 3. 准备时效性参数
        # ==========================
        duration_map = {}
        if mode == 'realtime':
            for l in req_levels: duration_map[l] = 0
        elif mode == 'fixed':
            fixed = self.soi_settings.get('fixed_duration', 30)
            for l in req_levels: duration_map[l] = fixed
        elif mode == 'hierarchical':
            custom_durs = self.soi_settings.get('level_durations', {})
            for l in req_levels: 
                duration_map[l] = custom_durs.get(l, 30)

        # ==========================
        # 3. 建立索引映射 & 读取 JSONL
        # ==========================
        total_frames = len(sequence.frames)
        # name_to_idx = {os.path.basename(p): i for i, p in enumerate(sequence.frames)}
        
        sparse_anno = {}

        # 逐行读取 JSONL 对于图片序列，lasot是从1开始 mgit是从0开始命名
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                info = json.loads(line)
                idx = info.get('frame_idx')
                sparse_anno[idx] = info

        # ==========================
        # 4. 核心循环：状态追踪与文本生成 (紧接在上面代码之后)
        # ==========================
        soi_list = []
        last_frame_final_text = None
        # 状态追踪器
        active_texts = {l: "" for l in req_levels}
        last_updates = {l: -99999 for l in req_levels}

        for i in range(total_frames):
            frame_anno = sparse_anno.get(i)
            
            # --- A. 更新阶段 (Inject) ---
            if frame_anno and frame_anno.get('status') != 'skip':
                text_content = frame_anno.get(lang_root_key, {})
                for l in req_levels:
                    key = f"level{l}"
                    raw_txt = text_content.get(key, '').strip()
                    if raw_txt:
                        if l in [1, 4]:
                            processed = raw_txt[0].upper() + raw_txt[1:]
                        else:
                            processed = raw_txt[0].lower() + raw_txt[1:]
                        active_texts[l] = processed
                        last_updates[l] = i
                    else:
                        # 如果新标注显式为空，表示该属性消失
                        active_texts[l] = ""
                        last_updates[l] = i

            # --- B. 检查过期 (Expire) ---
            valid_parts = []
            for l in req_levels:
                txt = active_texts[l]
                if txt:
                    elapsed = i - last_updates[l]
                    if elapsed <= duration_map[l]:
                        valid_parts.append(txt)
                    else:
                        # 过期清理
                        active_texts[l] = ""

            # --- C. 组装最终文本 (Assemble & Fallback) ---
            final_text = " ".join(valid_parts)
            
            # 如果拼出来的 SOI 文本是空的，使用原始文本回退
            if not final_text.strip():
                final_text = fallback_text
                
            # --- Change Detection ---
            # 如果是第0帧，或者文本跟上一帧不一样，则需要更新
            need_update = (i == 0) or (final_text != last_frame_final_text)
            
            last_frame_final_text = final_text

            soi_list.append({'text': final_text, 'need_update': need_update})

        return soi_list


################################################################################
# 
# 章节 2: 复制的基础数据集类 (按您的要求)
# 在这里粘贴 LaSOTDataset, VideoCubeDataset 和 TNL2kDataset 的完整代码。
#
################################################################################

class LaSOTDataset(BaseDataset):
    """
    LaSOT test set consisting of 280 videos (see Protocol-II in the LaSOT paper)

    注意: 这是一个被 SOIBenchmark '借用' 的辅助类。
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.lasot_path
        self.sequence_list = self._get_sequence_list_default() 
        self.clean_list = self.clean_seq_list()

    def clean_seq_list(self):
        clean_lst = []
        for i in range(len(self.sequence_list)):
            cls, _ = self.sequence_list[i].split('-')
            clean_lst.append(cls)
        return clean_lst

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        class_name = sequence_name.split('-')[0]
        anno_path = '{}/{}/{}/groundtruth.txt'.format(self.base_path, class_name, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        occlusion_label_path = '{}/{}/{}/full_occlusion.txt'.format(self.base_path, class_name, sequence_name)

        nlp_path = '{}/{}/{}/nlp.txt'.format(self.base_path, class_name, sequence_name)
        try:
            nlp_rect = load_text(str(nlp_path), delimiter=',', dtype=str)
            nlp_rect = str(nlp_rect)
        except:
            nlp_rect = '' # 容错

        full_occlusion = load_text(str(occlusion_label_path), delimiter=',', dtype=np.float64, backend='numpy')

        out_of_view_label_path = '{}/{}/{}/out_of_view.txt'.format(self.base_path, class_name, sequence_name)
        out_of_view = load_text(str(out_of_view_label_path), delimiter=',', dtype=np.float64, backend='numpy')

        target_visible = np.logical_and(full_occlusion == 0, out_of_view == 0)

        frames_path = '{}/{}/{}/img'.format(self.base_path, class_name, sequence_name)

        frames_list = ['{}/{:08d}.jpg'.format(frames_path, frame_number) for frame_number in range(1, ground_truth_rect.shape[0] + 1)]

        target_class = class_name
        return Sequence(sequence_name, frames_list, 'lasot', ground_truth_rect.reshape(-1, 4),
                        object_class=target_class, target_visible=target_visible, language_query=nlp_rect)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list_default(self):
        # 完整的 LaSOT 列表...
        sequence_list = []
        return sequence_list

    def _get_sequence_list(self):
        return self._get_sequence_list_default()


class VideoCubeDataset(BaseDataset):
    """
    VideoCube test set
    
    注意: 这是一个被 SOIBenchmark '借用' 的辅助类。
    """

    def __init__(self, split, version='tiny'):  # full
        super().__init__()

        self.split = split
        self.version = version

        json_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'videocube.json')
        try:
            f = open(json_path, 'r', encoding='utf-8')
            self.infos = json.load(f)[self.version]
            f.close()
        except FileNotFoundError:
            print(f"错误: 找不到 {json_path}。")
            print("请确保 'videocube.json' 文件与 'soi_benchmark.py' 位于同一目录 (lib/test/dataset/)。")
            raise
            
        self.sequence_list = self.infos[self.split]

        if split == 'test' or split == 'val':
            self.base_path = self.env_settings.videocube_path
        else:
            self.base_path = self.env_settings.videocube_path

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/{}/{}.txt'.format(self.base_path, 'attribute', 'groundtruth', sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        frames_path = r'{}/{}/{}/{}/{}_{}'.format(self.base_path, 'data', self.split, sequence_name, 'frame', sequence_name)
        
        frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith(".jpg")]
        frame_list.sort(key=lambda f: int(f[:-4]))
        frames_list = [os.path.join(frames_path, frame) for frame in frame_list]

        # (新增) 加载 MGIT/VideoCube 的 Story 文本描述
        # 假设 description 文件夹在数据集根目录下，文件名与序列名一致 (e.g., 001.json)
        desc_path = '{}/{}/{}/{}.json'.format(self.base_path, 'attribute', 'description', sequence_name)
        
        story_text = "" 
        
        if os.path.exists(desc_path):
            try:
                with open(desc_path, 'r', encoding='utf-8') as f:
                    desc_data = json.load(f)
                
                # 提取 story -> story_1 -> description
                # 注意：有些文件可能没有 story_1 或者结构稍有不同，这里做个容错
                if 'story' in desc_data and 'story_1' in desc_data['story']:
                    raw_text = desc_data['story']['story_1'].get('description', '')
                    if raw_text:
                        story_text = raw_text.strip()
                        # 确保首字母大写
                        story_text = story_text[0].upper() + story_text[1:]
                else:
                    # 如果没有 story，尝试读取 activity 或其他字段作为备选（可选）
                    print(f"Warning: No 'story_1' found in {sequence_name}.json")
                    
            except Exception as e:
                print(f"Error loading description for {sequence_name}: {e}")
        else:
            # 如果找不到描述文件，可能打印个警告或者保持为空
            # print(f"Warning: Description file not found: {desc_path}")
            pass

        return Sequence(sequence_name, frames_list, 'videocube_{}'.format(self.split), ground_truth_rect.reshape(-1, 4), language_query=story_text)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list_from_txt(self, split):
        path = r'{}/{}/{}_list.txt'.format(self.base_path, 'data', split)
        with open(path) as f:  # list.txt
            sequence_list = f.read().splitlines()
        return sequence_list


class TNL2kDataset(BaseDataset):
    """
    TNL2k test set

    注意: 这是一个被 SOIBenchmark '借用' 的辅助类。
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.tnl2k_path
        self.sequence_list = self._get_sequence_list_default()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        text_dsp_path = '{}/{}/language.txt'.format(self.base_path, sequence_name)
        text_dsp = load_str(text_dsp_path)

        frames_path = '{}/{}/imgs'.format(self.base_path, sequence_name)
        frames_list = [f for f in os.listdir(frames_path)]
        frames_list = sorted(frames_list)
        frames_list = ['{}/{}'.format(frames_path, frame_i) for frame_i in frames_list]

        return Sequence(sequence_name, frames_list, 'tnl2k', ground_truth_rect.reshape(-1, 4),
                        language_query=text_dsp)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list_default(self):
        sequence_list = []
        for seq in os.listdir(self.base_path):
            if os.path.isdir(os.path.join(self.base_path, seq)):
                sequence_list.append(seq)
        return sequence_list
    
    def _get_sequence_list(self):
        return self._get_sequence_list_default()
