from experiments.sentences import SentenceSelection, process
from dataset import get_examples
import argparse
ROPES_DATA_PATH = 'data/ropes/'

def main(args):
    if args.train:
        examples, _, _ = get_examples('train-v1.0.json')
    else:
        examples, _, _ = get_examples('dev-v1.0.json')
        
    process(examples, f'data/ropes/{args.file_name}.json', k=args.k, enforce_contains_answer=args.enforce_contains_answer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-name', type=str, required=True)
    parser.add_argument('--train', action="store_true")
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--enforce-contains-answer', action="store_true")

    args = parser.parse_args()
    print(args.train, args.k, args.file_name)
    main(args)
