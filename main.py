import torch
import argparse
from transformers import BertTokenizerFast, AutoModelForQuestionAnswering, AutoConfig, AdamW
from torch.utils.data import DataLoader, RandomSampler
from dataset import ROPES
import utils
from tqdm import tqdm

BERT_MODEL = "bert-base-cased"


def train(args, model, dataset, dev_dataset, tokenizer):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    train_sampler = RandomSampler(dataset)
    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=args.batch_size)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay
        },
        #{"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay":0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    model.zero_grad()
    epoch_iterator = tqdm(train_dataloader, desc="Iteration")
    best_em, best_f1 = -1, -1
    for i in range(args.epochs):
        for step, batch in enumerate(epoch_iterator):
            for t in batch:
                batch[t].to(device)
            #batch = tuple(t.to(args.device) for t in batch)

            # inputs = {
            #     "input_ids": batch[0],
            #     "attention_mask": batch[1],
            #     "token_type_ids": batch[2],
            #     "start_positions": batch[3],
            #     "end_positions": batch[4],
            # }

            outputs = model(**batch)
            loss = outputs[0]
            print(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            model.zero_grad()
        test(args, model, dev_dataset, tokenizer)

def test(args, model, dev_dataset, tokenizer):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    model.eval()
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size)
    dev_iterator = tqdm(dev_dataloader)
    total_loss = 0
    keys = ['input_ids', 'attention_mask', 'token_type_ids', 'start_positions', 'end_positions']
    predictions = {}
    for step, batch in enumerate(dev_iterator):
        for key in keys:
            batch[key].to(device)
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs[0]
            total_loss += loss.item()

        batch_size = batch['input_ids'].size(0)
        for i in range(batch_size):
            predictions[batch['qid'][i]] = tokenizer.convert_ids_to_tokens(batch['input_ids'][i])


def evaluate(predictions, tokenizer):
    pass




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--learning-rate', default=5e-5,type=float)
    parser.add_argument('--weight-decay', default=0.0, type=float)
    parser.add_argument('--adam_epsilon', default=1e-8, type=float)
    parser.add_argument('--max-grad-norm', default=1.0, type=float)
    args = parser.parse_args()
    
    config = AutoConfig.from_pretrained(BERT_MODEL)
    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL)
    model = AutoModelForQuestionAnswering.from_pretrained(BERT_MODEL, config=config)
    train_dataset = ROPES(tokenizer, 'train-v1.0.json')
    dev_dataset = ROPES(tokenizer, 'dev-v1.0.json')

    utils.set_random_seed(args.seed)
    train(args, model, train_dataset, dev_dataset, tokenizer)

if __name__ == '__main__':
    main()



