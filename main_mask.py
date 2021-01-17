import torch
import argparse
from transformers import (BertTokenizerFast, BertConfig, AdamW,
                          get_linear_schedule_with_warmup)
from model_mask import BertForQAandMLM
from torch.utils.data import DataLoader, RandomSampler
import torch.nn.functional as F
from dataset_mask import ROPES
import utils
import evaluate
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
BERT_MODEL = "bert-base-cased"


def train(args, model, dataset, dev_dataset, tokenizer, contrast_dataset=None):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    logger.info("***** Running training *****")
    logger.info("  Num train examples = %d", len(dataset))
    logger.info("  Num dev examples = %d", len(dev_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info(f"  Learning rate = {args.learning_rate}")
    logger.info(f"  Using device = {device}")
    logger.info(f"  Batch size = {args.batch_size}")
    logger.info(f"  Using random seed = {args.seed}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")

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
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps,
                                                num_training_steps=len(train_dataloader)*args.epochs)
    model.zero_grad()

    best_em, best_f1, contrast_em, contrast_f1 = -1, -1, -1, -1
    for i in range(args.epochs):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            for t in batch:
                batch[t] = batch[t].to(args.gpu)

            # qa
            batch["qa"] = True
            outputs = model(**batch)
            qa_loss = outputs[0]
            # mlm
            batch["qa"] = False
            outputs = model(**batch)
            mlm_loss = outputs[0]
            loss = (1 - args.mixing_ratio) * qa_loss + args.mixing_ratio * mlm_loss

            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

        dev_loss, dev_em, dev_f1 = test(args, model, dev_dataset, tokenizer)
        logger.info(f"***** EM:{dev_em} F1:{dev_f1} loss: {dev_loss} *****")

        model.train()
        if dev_em > best_em:
            best_em = dev_em
            best_f1 = dev_f1
            _, contrast_em, contrast_f1 = test(args, model, contrast_dataset, tokenizer, contrast=True)
            logger.info(f"***** Best Checkpoint EM:{best_em} F1:{best_f1}, "
                        f"Contrast EM:{contrast_em} F1:{contrast_f1} Saving... *****")
            checkpoint = {'args': args.__dict__, 'model': model.cpu()}
            torch.save(checkpoint, f"{args.output_dir}/best.pt")
            model.cuda()

    return best_em, best_f1, contrast_em, contrast_f1


def test(args, model, dev_dataset, tokenizer, contrast=False):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.dev_batch_size)
    dev_iterator = tqdm(dev_dataloader)
    total_loss = 0

    answers = {}
    for step, batch in enumerate(dev_iterator):

        qids = batch['id']
        del batch['id']
        for key in batch:
            batch[key] = batch[key].to(args.gpu)
        batch["qa"] = True

        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs[0]
            total_loss += loss.item()

        batch_size = batch['input_ids'].size(0)
        start_idxs, end_idxs = utils.discretize(F.softmax(outputs[1], dim=1), F.softmax(outputs[2], dim=1))
        for i in range(batch_size):
            s, e = start_idxs[i], end_idxs[i]
            answers[qids[i]] = tokenizer.decode(batch['input_ids'][i].tolist()[s:e + 1])

    # Compute EM and F1 on predictions
    res = evaluate.main(answers, tokenizer, contrast=contrast)
    return total_loss, res['exact_match'], res['f1']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="train/ropes10_0.5", type=str)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=10, type=int)
    parser.add_argument('--learning-rate', default=3e-5, type=float)
    parser.add_argument('--weight-decay', default=0.0, type=float)
    parser.add_argument('--adam_epsilon', default=1e-8, type=float)
    parser.add_argument('--max-grad-norm', default=1.0, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--dev-batch-size', default=8, type=int)
    parser.add_argument('--num-warmup-steps', default=0, type=int)
    parser.add_argument('--mixing_ratio', default=0.5, type=float)
    parser.add_argument('--find_mask', default="label", type=str)
    parser.add_argument('--gradient_accumulation_steps', default=2, type=int)
    args = parser.parse_args()

    utils.set_random_seed(args.seed)
    config = BertConfig.from_pretrained(BERT_MODEL, cache_dir='./cache')
    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL, cache_dir='./cache')
    model = BertForQAandMLM.from_pretrained(BERT_MODEL, config=config, cache_dir='./cache')
    train_dataset = ROPES(tokenizer, 'ropes_no_coref_train_combined_examples.json', find_mask=args.find_mask)
    dev_dataset = ROPES(tokenizer, 'ropes_no_coref_dev_combined_examples.json', eval=True, find_mask=args.find_mask)
    contrast_dataset = ROPES(tokenizer, 'ropes_no_coref_contrast_examples.json', eval=True, find_mask=args.find_mask)

    em, f1, contrast_em, contrast_f1 = train(args, model, train_dataset, dev_dataset, tokenizer,
                                             contrast_dataset=contrast_dataset)

    with open("train/log", 'a') as f:
        f.write(f'\n{args.output_dir}, {em}, {f1}, {contrast_em}, {contrast_f1}\n')


if __name__ == '__main__':
    main()
