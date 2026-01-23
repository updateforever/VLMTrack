# SUTrack 开发笔记

## 2026-01-23: 添加速度分析脚本

### 任务背景
评测代码部分对于指标分析只有AUC等精度指标，没有对速度进行统计。但推理时已经保存了 `time.txt` 文件，需要补充速度分析功能。

### 实现内容

#### 1. 新增文件

**`lib/test/analysis/speed_analysis.py`** - 速度分析核心模块
- `load_time_file()`: 加载 `{seq_name}_time.txt` 文件
- `extract_speed_results()`: 从所有序列提取时间数据
- `compute_speed_statistics()`: 计算速度统计指标
  - 平均 FPS (序列级)
  - 整体 FPS (帧级)
  - FPS 标准差、最小值、最大值、中位数
  - 平均帧处理时间 (ms)
- `print_speed_results()`: 打印详细速度报告
- `print_speed_comparison()`: 打印简化对比表
- `get_per_sequence_fps()`: 获取每序列FPS用于自定义分析

**`tracking/analysis_speed.py`** - 速度分析入口脚本
- 类似 `analysis_results.py` 的使用方式
- 配置 trackers 和 dataset 后直接运行

#### 2. 时间文件格式 (参考 running.py)

保存路径: `{results_dir}/{dataset}/{seq_name}_time.txt`

每行一个浮点数，表示对应帧的处理时间（秒）。

#### 3. 使用方法

```python
# 方法1: 使用入口脚本
# 编辑 tracking/analysis_speed.py 配置 trackers 和 dataset_name
python tracking/analysis_speed.py

# 方法2: 在代码中调用
from lib.test.analysis.speed_analysis import print_speed_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = trackerlist(name='sutrack', parameter_name='sutrack_b224', 
                       dataset_name='lasot', display_name='SUTrack-B224')
dataset = get_dataset('lasot')
print_speed_results(trackers, dataset, report_name='lasot_speed')
```

#### 4. 输出示例

```
===========================================================================
Speed Analysis
===========================================================================
Tracker           | Avg FPS      | Std FPS      | Overall FPS  | Avg Time(ms) |
---------------------------------------------------------------------------
SUTrack-B224      | 45.23        | 5.67         | 44.89        | 22.28        |
===========================================================================
```

### 关键设计决策
1. **排除首帧**: 默认排除第一帧时间（初始化帧通常较慢）
2. **跳过缺失**: 默认跳过没有time.txt的序列
3. **两级FPS统计**: 
   - `Avg FPS`: 每序列FPS的平均值（序列级均值）
   - `Overall FPS`: 总帧数/总时间（帧级均值）

---
