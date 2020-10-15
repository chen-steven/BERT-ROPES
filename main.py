import torch
import argparse
from transformers import (BertTokenizerFast, AutoModelForQuestionAnswering, AutoConfig, AdamW,
                          get_linear_schedule_with_warmup)
from torch.utils.data import DataLoader, RandomSampler
import torch.nn.functional as F
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
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay":0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps,
                                                num_training_steps=len(train_dataloader)*args.epochs)
    model.zero_grad()
    #epoch_iterator = tqdm(train_dataloader, desc="Iteration")
    best_em, best_f1 = -1, -1
    for i in range(args.epochs):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            for t in batch:
                batch[t] = batch[t].to(args.gpu)

            outputs = model(**batch)
            loss = outputs[0]
            #print(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()
            model.zero_grad()
        dev_loss, dev_em, dev_f1 = test(args, model, dev_dataset, tokenizer)
        model.train()
        if dev_em > best_em:
            best_em = dev_em
        print('EM', dev_em, 'F1', dev_f1, 'loss',dev_loss)


def test(args, model, dev_dataset, tokenizer):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.dev_batch_size)
    dev_iterator = tqdm(dev_dataloader)
    total_loss = 0
    keys = ['input_ids', 'attention_mask', 'token_type_ids', 'start_positions', 'end_positions']
    predictions = {}
    answers = {}
    for step, batch in enumerate(dev_iterator):
     #   print(batch)
        qids = batch['id']
        del batch['id']
        for key in batch:
            batch[key] = batch[key].to(args.gpu)
#        print(batch)
        #batch['return_dict'] = True
        #qids = batch['id']
        #del batch['id']
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs[0]
            total_loss += loss.item()

        batch_size = batch['input_ids'].size(0)
        start_idxs, end_idxs = utils.discretize(F.softmax(outputs[1], dim=1), F.softmax(outputs[2], dim=1))
        for i in range(batch_size):
            s, e = start_idxs[i], end_idxs[i]
            answers[qids[i]] = tokenizer.decode(batch['input_ids'][i][s:e+1])
            # predictions[qids[i]] = evaluate.ROPESResult(qids[i],
            #                                            outputs['start_logits'][i],
            #                                            outputs['end_logits'][i],
            #                                            batch['input_ids'][i])
        

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
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--dev-batch-size', default=32, type=int)
    parser.add_argument('--num-warmup-steps', default=0, type=int)
    args = parser.parse_args()
    
    config = AutoConfig.from_pretrained(BERT_MODEL)
    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL)
    model = AutoModelForQuestionAnswering.from_pretrained(BERT_MODEL, config=config)
    train_dataset = ROPES(tokenizer, 'train-v1.0.json')
    dev_dataset = ROPES(tokenizer, 'dev-v1.0.json', eval=True)

    utils.set_random_seed(args.seed)
#    print(test(args, model, dev_dataset, tokenizer))
    train(args, model, train_dataset, dev_dataset, tokenizer)

if __name__ == '__main__':
    main()



