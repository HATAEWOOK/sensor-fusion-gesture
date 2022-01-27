import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", default='None', help='train, evaluation')
    