from lib.test.evaluation.environment import EnvSettings
import os


def local_env_settings():
    settings = EnvSettings()

    # 自动解析工程根路径（.../lib/test/evaluation/local.py -> repo root）
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

    # 固定数据根目录（本机）
    data_root = '/root/user-data/PUBLIC_DATASETS'

    # Set your local paths here.
    settings.davis_dir = ''
    settings.got10k_lmdb_path = f'{data_root}/got10k_lmdb'
    settings.got10k_path = f'{data_root}/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_extension_subset_path = f'{data_root}/lasot_extension_subset'
    settings.lasot_lmdb_path = f'{data_root}/lasot_lmdb'

    settings.network_path = os.path.join(repo_root, 'test/networks')  # Where tracking networks are stored.
    settings.nfs_path = f'{data_root}/nfs'
    settings.otb_path = f'{data_root}/OTB2015'
    settings.otblang_path = f'{data_root}/otb_lang'
    settings.prj_dir = repo_root
    settings.result_plot_path = os.path.join(repo_root, 'test/result_plots')
    settings.results_path = os.path.join(repo_root, 'test/tracking_results')  # Where to store tracking results
    settings.save_dir = repo_root
    settings.segmentation_path = os.path.join(repo_root, 'test/segmentation_results')
    settings.tc128_path = f'{data_root}/TC128'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = f'{data_root}/trackingnet'
    settings.uav_path = f'{data_root}/UAV123'
    settings.vot_path = f'{data_root}/VOT2019'
    settings.youtubevos_dir = ''

    # 固定测试数据集路径（不再通过环境变量和自动探测）
    settings.lasot_path = f'{data_root}/lasot'
    settings.lasotlang_path = settings.lasot_path
    settings.videocube_path = f'{data_root}/MGIT'
    settings.tnl2k_path = f'{data_root}/TNL2K/TNL2K_test_subset'
    settings.soi_bench_path = f'{data_root}/SOIBench/test_anno'

    # 固定关键帧索引路径
    settings.keyframe_root = f'{data_root}/SOIBench/KeyFrame/scene_changes_resnet/top_10'

    return settings
