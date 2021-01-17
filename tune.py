import  os


def tune():
    for label in ["label"]:
        for mixing_ratio in [0.7, 0.9]:
            os.system(f"mkdir train/ropes10_{mixing_ratio}_{label}")
            os.system(f"python main_mask.py --output_dir train/ropes10_{mixing_ratio}_{label} "
                      f"--mixing_ratio {mixing_ratio} --find_mask {label} ")


def tune_mc():
    for label in ["label1", "label2"]:
        for mixing_ratio in [0.1, 0.3, 0.5]:
            os.system(f"mkdir train/ropes10_{mixing_ratio}_{label}_mc")
            os.system(f"python main_mask_mc.py --output_dir train/ropes10_{mixing_ratio}_{label}_mc "
                      f"--mixing_ratio {mixing_ratio} --find_mask {label} ")


if __name__ == '__main__':
    tune_mc()