from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/data/wyp/VLMTrack//data/got10k_lmdb'
    settings.got10k_path = '/data/wyp/VLMTrack//data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_extension_subset_path = '/data/wyp/VLMTrack//data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/data/wyp/VLMTrack//data/lasot_lmdb'

    settings.network_path = '/data/wyp/VLMTrack//test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/data/wyp/VLMTrack//data/nfs'
    settings.otb_path = '/data/wyp/VLMTrack//data/OTB2015'
    settings.otblang_path = '/data/wyp/VLMTrack//data/otb_lang'
    settings.prj_dir = '/data/wyp/VLMTrack/'
    settings.result_plot_path = '/data/wyp/VLMTrack//test/result_plots'
    settings.results_path = '/data/wyp/VLMTrack//test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/data/wyp/VLMTrack/'
    settings.segmentation_path = '/data/wyp/VLMTrack//test/segmentation_results'
    settings.tc128_path = '/data/wyp/VLMTrack//data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/member/data1/DATASETS_PUBLIC/TNL2K/TNL2K_CVPR2021/test'
    settings.tpl_path = ''
    settings.trackingnet_path = '/data/wyp/VLMTrack//data/trackingnet'
    settings.uav_path = '/data/wyp/VLMTrack//data/UAV123'
    settings.vot_path = '/data/wyp/VLMTrack//data/VOT2019'
    settings.youtubevos_dir = ''

    settings.lasot_path = '/data/DATASETS_PUBLIC/lasot'
    settings.lasotlang_path = '/data/DATASETS_PUBLIC/lasot'
    settings.videocube_path = '/data/DATASETS_PUBLIC/MGIT'
    settings.tnl2k_path = '/data/DATASETS_PUBLIC/TNL2K/TNL2K_test_subset'
    settings.soi_bench_path = '/data/DATASETS_PUBLIC/SOIBench/test_anno'

    # =========== Keyframe paths for different datasets ===========
    # 场景变化检测结果路径,用于稀疏跟踪
    # lib/test/evaluation/local.py
    settings.keyframe_root = {
        'root': '/data/DATASETS_PUBLIC/SOIBench_f',
        'model': 'scene_changes_resnet',
        'threshold': 'top_10',
    }





    return settings

