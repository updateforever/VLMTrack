import os

import numpy as np
from lib.test.evaluation.data import Sequence,VideoCude_Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import json
import pandas as pd

############
# current 00000492.png of test_015_Sord_video_Q01_done is damaged and replaced by a copy of 00000491.png
############


class VideoCubeDataset(BaseDataset):
    """
    VideoCube test set
    """

    def __init__(self, split, version='full'):
        super().__init__()

        self.split = split
        self.version = version

        f = open(os.path.join(os.path.split(os.path.realpath(__file__))[0], 'videocube.json'), 'r', encoding='utf-8')
        self.infos = json.load(f)[self.version]
        f.close()

        # print('sequence_list')

        self.sequence_list = self.infos[self.split]

        print('sequence_list', self.sequence_list)

        # 数据集路径统一在 local.py 中配置
        self.base_path = self.env_settings.videocube_path

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    # def _construct_sequence(self, sequence_name):
    #     # class_name = sequence_name.split('-')[0]
    #     anno_path = '{}/{}/{}/{}.txt'.format(self.base_path, 'attribute', 'groundtruth', sequence_name)

    #     ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

    #     # text_dsp_path = '{}/{}/language.txt'.format(self.base_path, sequence_name)
    #     # text_dsp = load_str(text_dsp_path)

    #     nlp_path = self.base_path+'/VideoCube_NL/02-activity&story/{}.xlsx'.format(sequence_name)
    #     nlp_tab = pd.read_excel(nlp_path)
    #     nlp_rect = nlp_tab.iloc[:, [14]].values
    #     nlp_rect = nlp_rect[-1, 0]
    #     # print('nlp_rect', nlp_rect)

    #     frames_path = r'{}/{}/{}/{}/{}_{}'.format(self.base_path, 'data', self.split, sequence_name, 'frame', sequence_name)
    #     # frames_path = frames_path.replace('\\', '')
    #     frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith(".jpg")]
    #     frame_list.sort(key=lambda f: int(f[:-4]))
    #     frames_list = [os.path.join(frames_path, frame) for frame in frame_list]

    #     # target_class = class_name
    #     return VideoCude_Sequence(sequence_name, frames_list, 'videocube', ground_truth_rect.reshape(-1, 4), object_class=None, target_visible=None, language_query=nlp_rect)

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

    def _get_sequence_list(self, split):
        path = r'{}/{}/{}_list.txt'.format(self.base_path, 'data', split)
        # path = path.replace('\\', '')
        with open(path) as f:  # list.txt
            sequence_list = f.read().splitlines()

        return sequence_list
