import os


class Constant:
    DEBUG = False
    BASE_DIR = os.path.join(os.path.dirname(__file__), '../')
    ASSET_DIR = os.path.join(BASE_DIR, 'asset')
    BIT_FEATURES_CSV = os.path.join(ASSET_DIR, 'bit_features.csv')
    ENGINEER_FEATURES_CSV = os.path.join(ASSET_DIR, 'engineer_features.csv')
    DATASET_CONFIGURATION_CSV = os.path.join(
        ASSET_DIR,
        'dataset_configuration_debug.csv') if DEBUG else os.path.join(
            ASSET_DIR, 'dataset_configuration.csv')

    BERT_PREPROCESS_MODEL = os.path.join(ASSET_DIR, 'bert_zh_preprocess_3')
    BERT_CHECKPOINT = os.path.join(
        ASSET_DIR, 'chinese_L-12_H-768_A-12/tmp/temp_dir/raw/',
        'bert_model.ckpt')


if __name__ == '__main__':
    print(Constant.BIT_FEATURES_CSV)
