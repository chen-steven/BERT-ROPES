import torch
import argparse
from transformers import (BertTokenizerFast, AutoModelForQuestionAnswering, AutoConfig, AdamW,
                          get_linear_schedule_with_warmup)
from torch.utils.data import DataLoader, RandomSampler
import torch.nn.functional as F
from dataset import ROPES, MaskedSentenceRopes
import utils
import evaluate
from tqdm import tqdm

BERT_MODEL = "bert-base-cased"


def train(args, model, dataset, dev_dataset, tokenizer):
    if args.use_multi_labels:
        print('Using multiple labels')
    if args.use_augmented_examples:
        print('Using augmented examples')
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
    utils.set_random_seed(args.seed)
    model.zero_grad()
    #epoch_iterator = tqdm(train_dataloader, desc="Iteration")
    best_em, best_f1 = -1, -1
    for i in range(args.epochs):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            if args.use_multi_labels:
                start_labels = batch['start_multi_label']
                end_labels = batch['end_multi_label']
                del batch['start_multi_label']
                del batch['end_multi_label']

            for t in batch:
                batch[t] = batch[t].to(args.gpu)

            outputs = model(**batch)

            if args.use_multi_labels:
                nll_loss = 0
                start_logits, end_logits = outputs[1], outputs[2]
                batch_size = start_logits.size(0)
            
                for j in range(batch_size):
                
                    #for s0, s1, e0, e1 in zip(start_labels[j], end_labels[j]):
                    #    if s != -1 and e != -1:
                    c = 1
                    tmp = utils.cross_entropy_loss(start_logits[j], start_labels[0][j])+utils.cross_entropy_loss(end_logits[j], end_labels[0][j])
                    if start_labels[1][j] != -1 and end_labels[1][j] != -1:
                        c = 2
                        tmp += utils.cross_entropy_loss(start_logits[j], start_labels[1][j])+utils.cross_entropy_loss(end_logits[j], end_labels[1][j])
                    
                    nll_loss += tmp/c
                loss = nll_loss/batch_size
            else:
                loss = outputs[0]
            
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
    return best_em

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
            answers[qids[i]] = tokenizer.decode(batch['input_ids'][i].tolist()[s:e+1])
            # predictions[qids[i]] = evaluate.ROPESResult(qids[i],
            #                                            outputs['start_logits'][i],
            #                                            outputs['end_logits'][i],
            #                                            batch['input_ids'][i])
        

    res = evaluate.main(answers, tokenizer)
    return total_loss, res['exact_match'], res['f1']

def trainer(args, model, dataset, dev_dataset, tokenizer):
    seeds = [10, 20, 30]
    ems = []
    for seed in seeds:
        print(f'Running with seed: {seed}')
        args.seed = seed
        utils.set_random_seed(seed)
        best_em = train(args, model, dataset, dev_dataset, tokenizer)
        ems.append(best_em)
    print(sum(ems)/len(ems))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--learning-rate', default=1e-5,type=float)
    parser.add_argument('--weight-decay', default=0.0, type=float)
    parser.add_argument('--adam_epsilon', default=1e-8, type=float)
    parser.add_argument('--max-grad-norm', default=1.0, type=float)
    parser.add_argument('--epochs', default=4, type=int)
    parser.add_argument('--dev-batch-size', default=32, type=int)
    parser.add_argument('--num-warmup-steps', default=0, type=int)
    parser.add_argument('--use-augmented-examples', action="store_true")
    parser.add_argument('--use-multi-labels', action="store_true")
    parser.add_argument('--use-masked-sentences', action="store_true")
    args = parser.parse_args()
    
    config = AutoConfig.from_pretrained(BERT_MODEL)
    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL)
    model = AutoModelForQuestionAnswering.from_pretrained(BERT_MODEL, config=config)

    if not args.use_masked_sentences:
        train_dataset = ROPES(args, tokenizer, 'train-v1.0.json')
        dev_dataset = ROPES(args, tokenizer, 'dev-v1.0.json', eval=True)
    else:
        train_dataset = MaskedSentenceRopes(args, tokenizer, 'train-top-sentences-contains-answer.json')
        dev_dataset = MaskedSentenceRopes(args, tokenizer, 'dev-top-sentences-contains-answer.json', eval=True)

    utils.set_random_seed(args.seed)
#    print(test(args, model, dev_dataset, tokenizer))
    trainer(args, model, train_dataset, dev_dataset, tokenizer)

if __name__ == '__main__':
    main()



