import torch
import argparse
from transformers import BertTokenizerFast, AutoModelForQuestionAnswering, AutoConfig, AdamW
from torch.utils.data import DataLoader, RandomSampler
from dataset import ROPES
import utils
import evaluate
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
                batch[t] = batch[t].to(args.gpu)

            outputs = model(**batch)
            loss = outputs[0]
            #print(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            model.zero_grad()
        test(args, model, dev_dataset, tokenizer)

def test(args, model, dev_dataset, tokenizer):
    print('hello')
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    model.eval()
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size)
    dev_iterator = tqdm(dev_dataloader)
    total_loss = 0
    keys = ['input_ids', 'attention_mask', 'token_type_ids', 'start_positions', 'end_positions']
    predictions = {}
    for step, batch in enumerate(dev_iterator):
        for key in keys:
            batch[key] = batch[key].to(device)
        batch['return_dict'] = True
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs['loss']
            total_loss += loss.item()

        batch_size = batch['input_ids'].size(0)
        for i in range(batch_size):
            predictions[batch['qid'][i]] = evaluate.ROPESResult(batch['quid'][i],
                                                       outputs['start_logits'][i],
                                                       outputs['end_logits'][i],
                                                       batch['input_ids'][i])

    res = evaluate.main(predictions, tokenizer)
    return total_loss, res['exact_match'], res['f1']






def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--learning-rate', default=5e-5,type=float)
    parser.add_argument('--weight-decay', default=0.0, type=float)
    parser.add_argument('--adam_epsilon', default=1e-8, type=float)
    parser.add_argument('--max-grad-norm', default=1.0, type=float)
    parser.add_argument('--epochs', default=4, type=int)
    args = parser.parse_args()
    
    config = AutoConfig.from_pretrained(BERT_MODEL)
    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL)
    model = AutoModelForQuestionAnswering.from_pretrained(BERT_MODEL, config=config)
    print('tesasdfadf')
    train_dataset = ROPES(tokenizer, 'train-v1.0.json')
    dev_dataset = ROPES(tokenizer, 'dev-v1.0.json', eval=True)
    print(dev_dataset)

    utils.set_random_seed(args.seed)
    #test(args, model, dev_dataset, tokenizer)
    train(args, model, train_dataset, dev_dataset, tokenizer)

if __name__ == '__main__':
    main()



