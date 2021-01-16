import  os


def tune():
    for mixing_ratio in [0.0, 0.1, 0.3, 0.5]:
        os.system(f"mkdir train/ropes10_{mixing_ratio}")
        os.system(f"python main_mask.py --output_dir train/ropes10_{mixing_ratio} "
                  f"--mixing_ratio {mixing_ratio} ")


if __name__ == '__main__':
    tune()