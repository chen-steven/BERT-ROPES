from experiments.sentences import SentenceSelection, process
from dataset import get_examples
ROPES_DATA_PATH = 'data/ropes/'

def main():
    examples, _, _ = get_examples('dev-v1.0.json')
    process(examples, 'data/ropes/masked_dev.json', k=5)


if __name__ == '__main__':
    main()