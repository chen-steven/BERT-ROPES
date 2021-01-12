import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, AutoModelForQuestionAnswering, AutoConfig
from tqdm import tqdm
import utils
import torch.nn.functional as F
from dataset import ROPES
import evaluate
import argparse
import json

BERT_MODEL = 'bert-base-cased'

def create_choices(args, model, dev_dataset, tokenizer, split='train'):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size)
    dev_iterator = tqdm(dev_dataloader)
    total_loss = 0
    keys = ['input_ids', 'attention_mask', 'token_type_ids', 'start_positions', 'end_positions']
    predictions = {}
    candidate_answers = {}
    answers = {}
    for step, batch in enumerate(dev_iterator):

        qids = batch['id']
        del batch['id']
        for key in batch:
            batch[key] = batch[key].to(args.gpu)

        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs[0]
            total_loss += loss.item()

            start_probs = torch.softmax(outputs[1], dim=-1)
            sorted_start_probs, start_indices = torch.sort(start_probs, dim=-1)
            end_probs = torch.softmax(outputs[2], dim=-1)
            sorted_end_probs, end_indices = torch.sort(end_probs, dim=-1)
            

        batch_size = batch['input_ids'].size(0)
        start_idxs, end_idxs = utils.discretize(F.softmax(outputs[1], dim=1), F.softmax(outputs[2], dim=1))
        
        choice_start_idxs, choice_end_idxs = utils.discretize(sorted_start_probs[:,:-1], sorted_end_probs[:,:-1])
        choice_start_idxs1, choice_end_idxs1 = utils.discretize(sorted_start_probs[:, :-2], sorted_end_probs[:, :-2])

        for i in range(batch_size):
            s, e = start_idxs[i], end_idxs[i]
            answers[qids[i]] = tokenizer.decode(batch['input_ids'][i].tolist()[s:e + 1])

            choice_s, choice_e = start_indices[i][choice_start_idxs[i]], end_indices[i][choice_end_idxs[i]]
            choice_e = min(choice_e, choice_s + 5)
            choice_s1, choice_e1 = start_indices[i][choice_start_idxs1[i]], end_indices[i][choice_end_idxs1[i]]
            choice_e1 = min(choice_e1, choice_s1 + 5)
            ans_choice = tokenizer.decode(batch['input_ids'][i].tolist()[choice_s: choice_e+1])
            
            ans_choice1 = tokenizer.decode(batch['input_ids'][i].tolist()[choice_s1: choice_e1+1])
            if step%100==0:
                print(ans_choice,ans_choice1, answers[qids[i]])
            candidate_answers[qids[i]] = [ans_choice1, ans_choice, answers[qids[i]]]


    res = evaluate.main(answers, tokenizer, split=split)
    return candidate_answers, res['exact_match'], res['f1']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--split', type=str)
    args = parser.parse_args()


    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL)
    config = AutoConfig.from_pretrained(BERT_MODEL)
    model = AutoModelForQuestionAnswering.from_pretrained(BERT_MODEL, config=config)

    dev_dataset = ROPES(tokenizer, '../../'+evaluate.SPLIT_MAP[args.split], eval=True)

    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    candidate_answers, em, f1 = create_choices(args, model, dev_dataset, tokenizer, split=args.split)
    print(len(candidate_answers))
    print(em, f1)

    json.dump(candidate_answers, open(f'{args.split}_candidate_answers.json', 'w'))
    

