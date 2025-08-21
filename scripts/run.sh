python main_v2.py --config config/config.yaml --video_data_save_dir data --device cuda:0

# Extract features
python main_extract_features.py --config config/config.yaml --save_dir data --device cuda:0

# v2
# python main_extract_features.py --config config/config.yaml --save_dir data --subset_list data/subset0.txt --device cuda:0
# python main_extract_features.py --config config/config.yaml --save_dir data --subset_list data/subset1.txt --device cuda:0
# python main_extract_features.py --config config/config.yaml --save_dir data --subset_list data/subset2.txt --device cuda:0
# python main_extract_features.py --config config/config.yaml --save_dir data --subset_list data/subset3.txt --device cuda:0
# python main_extract_features.py --config config/config.yaml --save_dir data --subset_list data/subset4.txt --device cuda:0
# python main_extract_features.py --config config/config.yaml --save_dir data --subset_list data/subset5.txt --device cuda:0
# python main_extract_features.py --config config/config.yaml --save_dir data --subset_list data/subset6.txt --device cuda:0
# python main_extract_features.py --config config/config.yaml --save_dir data --subset_list data/subset7.txt --device cuda:0
# python main_extract_features.py --config config/config.yaml --save_dir data --subset_list data/subset8.txt --device cuda:0
python main_extract_features.py --config config/config.yaml --save_dir data --subset_list data/subset9.txt --device cuda:7
python main_extract_features.py --config config/config.yaml --save_dir data --subset_list data/subset10.txt --device cuda:7
# ---- Not run
python main_extract_features.py --config config/config.yaml --save_dir data --subset_list data/subset11.txt --device cuda:7

# -- 12 dang running
# python main_extract_features.py --config config/config.yaml --save_dir data --subset_list data/subset12.txt --device cuda:7
python main_extract_features.py --config config/config.yaml --save_dir data --subset_list data/subset13.txt --device cuda:7
# ----
python main_extract_features.py --config config/config.yaml --save_dir data --subset_list data/subset14.txt --device cuda:0
python main_extract_features.py --config config/config.yaml --save_dir data --subset_list data/subset15.txt --device cuda:0
python main_extract_features.py --config config/config.yaml --save_dir data --subset_list data/subset16.txt --device cuda:0
python main_extract_features.py --config config/config.yaml --save_dir data --subset_list data/subset17.txt --device cuda:0
python main_extract_features.py --config config/config.yaml --save_dir data --subset_list data/subset18.txt --device cuda:0