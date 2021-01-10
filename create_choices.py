import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, AutoModelForQuestionAnswering, AutoConfig
from tqdm import tqdm
import utils
import torch.nn.functional as F
from dataset import ROPES
import evaluate
import argparse

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

            logits = outputs[1]
            sorted_logits, indices = torch.sort(logits, dim=-1)

        batch_size = batch['input_ids'].size(0)
        start_idxs, end_idxs = utils.discretize(F.softmax(outputs[1], dim=1), F.softmax(outputs[2], dim=1))
        for i in range(batch_size):
            s, e = start_idxs[i], end_idxs[i]
            answers[qids[i]] = tokenizer.decode(batch['input_ids'][i].tolist()[s:e + 1])


    res = evaluate.main(answers, tokenizer, split=split)
    return total_loss, res['exact_match'], res['f1']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default="cuda:0", type=str)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--checkpoint', type=str)
    args = parser.parse_args()


    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL)
    config = AutoConfig.from_pretrained(BERT_MODEL)
    model = AutoModelForQuestionAnswering.from_pretrained(BERT_MODEL, config=config)

    dev_dataset = ROPES(tokenizer, 'dev-v1.0.json', eval=True)

    model.load_state_dict(torch.load(args.checkpoint, map_location="gpu"))
    em, f1 = create_choices(args, model, dev_dataset, tokenizer, split="dev")

