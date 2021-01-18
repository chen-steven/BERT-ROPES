import  os


def tune():
    for label in ["label"]:
        for mixing_ratio in [0.7, 0.9]:
            os.system(f"mkdir train/ropes10_{mixing_ratio}_{label}")
            os.system(f"python main_mask.py --output_dir train/ropes10_{mixing_ratio}_{label} "
                      f"--mixing_ratio {mixing_ratio} --find_mask {label} ")


def tune_mc():
    for label in ["label"]:
        for mixing_ratio in [0.0, 0.1, 0.3, 0.5]:
            os.system(f"mkdir train/ropes10_{mixing_ratio}_{label}_mc")
            os.system(f"python main_mask_mc.py --output_dir train/ropes10_{mixing_ratio}_{label}_mc "
                      f"--mixing_ratio {mixing_ratio} --find_mask {label} ")


def tune_mc_all():
    for label in ["label2"]:
        os.system(f"mkdir train/ropes10_{label}_mc_all1")
        os.system(f"python main_mask_mc_all.py --output_dir train/ropes10_{label}_mc_all1 "
                  f" --find_mask {label} ")


if __name__ == '__main__':
    tune_mc_all()