from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/member/data2/wyp/SOT/VLMTrack/data/got10k_lmdb'
    settings.got10k_path = '/home/member/data2/wyp/SOT/VLMTrack/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_extension_subset_path = '/home/member/data2/wyp/SOT/VLMTrack/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home/member/data2/wyp/SOT/VLMTrack/data/lasot_lmdb'

    settings.network_path = '/home/member/data2/wyp/SOT/VLMTrack/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/member/data2/wyp/SOT/VLMTrack/data/nfs'
    settings.otb_path = '/home/member/data2/wyp/SOT/VLMTrack/data/OTB2015'
    settings.otblang_path = '/home/member/data2/wyp/SOT/VLMTrack/data/otb_lang'
    settings.prj_dir = '/home/member/data2/wyp/SOT/VLMTrack'
    settings.result_plot_path = '/home/member/data2/wyp/SOT/VLMTrack/test/result_plots'
    settings.results_path = '/home/member/data2/wyp/SOT/VLMTrack/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/member/data2/wyp/SOT/VLMTrack'
    settings.segmentation_path = '/home/member/data2/wyp/SOT/VLMTrack/test/segmentation_results'
    settings.tc128_path = '/home/member/data2/wyp/SOT/VLMTrack/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/member/data1/DATASETS_PUBLIC/TNL2K/TNL2K_CVPR2021/test'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/member/data2/wyp/SOT/VLMTrack/data/trackingnet'
    settings.uav_path = '/home/member/data2/wyp/SOT/VLMTrack/data/UAV123'
    settings.vot_path = '/home/member/data2/wyp/SOT/VLMTrack/data/VOT2019'
    settings.youtubevos_dir = ''

    settings.lasot_path = '/home/member/data1/DATASETS_PUBLIC/LaSOT/LaSOTBenchmark'
    settings.lasotlang_path = '/home/member/data2/wyp/SOT/VLMTrack/data/lasot'
    settings.videocube_path = '/home/member/data1/DATASETS_PUBLIC/MGIT/VideoCube/MGIT-Test'
    settings.lasotlang_path = '/home/member/data1/DATASETS_PUBLIC/LaSOT/LaSOTBenchmark'

    return settings

