
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_size", type=int, default=128)
    return parser.parse_args()
    