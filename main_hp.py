import torch
import argparse
from transformers import (BertTokenizerFast, RobertaTokenizerFast, AutoModelForQuestionAnswering, AutoConfig, AdamW,
                          get_linear_schedule_with_warmup)
from torch.utils.data import DataLoader, RandomSampler
import torch.nn.functional as F
from dataset_hp import HOTPOT
import utils
import evaluate_hp
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# BERT_MODEL = "bert-base-cased"
BERT_MODEL = "roberta-large"


def train(args, model, dataset, dev_dataset, adv_dev_dataset, tokenizer):
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
    logger.info(f"  Effective batch size = {args.batch_size * args.gradient_accumulation_steps}")

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

    best_em, best_f1 = -1, -1
    for i in range(args.epochs):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            for t in batch:
                batch[t] = batch[t].to(args.gpu)

            input_batch = {t: batch[t] for t in batch if t not in ["start_labels", "end_labels"]}
            outputs = model(**input_batch)

            if args.binary:
                start_logits, end_logits = outputs[1], outputs[2]
                loss = binary_cross_entropy(start_logits, batch["start_labels"], mask=input_batch["attention_mask"]) + \
                       binary_cross_entropy(end_logits, batch["end_labels"], mask=input_batch["attention_mask"])
            else:
                loss = outputs[0]

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

        dev_loss, dev_em, dev_f1 = test(args, model, dev_dataset, tokenizer)
        logger.info(f"***** Evaluation for epoch {i + 1} *****")
        logger.info(f"EM: {dev_em}, F1: {dev_f1}, loss: {dev_loss}")

        model.train()
        if dev_em > best_em:
            best_em = dev_em
            best_f1 = dev_f1
            adv_dev_loss, adv_dev_em, adv_dev_f1 = test(args, model, adv_dev_dataset, tokenizer, adv=True)
            logger.info(f"adv EM: {adv_dev_em}, adv F1: {adv_dev_f1}, adv loss: {adv_dev_loss}")
            logger.info(f"***** Best Checkpoint, Saving... *****")
            checkpoint = {'args': args.__dict__, 'model': model.cpu()}
            torch.save(checkpoint, f"{args.output_dir}/best.pt")
            model.cuda()

    return best_em, best_f1


def binary_cross_entropy(logits, p1, mask=None):
    p2 = torch.sigmoid(logits)
    if mask is not None:
        p2 = p2[mask.bool()]
        p1 = p1[mask.bool()]
    return -torch.mean(p1 * torch.log(p2 + 1e-30) + (1 - p1) * torch.log(1 - p2 + 1e-30))


def test(args, model, dev_dataset, tokenizer, adv=False):
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

        with torch.no_grad():
            input_batch = {t: batch[t] for t in batch if t not in ["start_labels", "end_labels"]}
            outputs = model(**input_batch)

            if args.binary:
                start_logits, end_logits = outputs[1], outputs[2]
                loss = binary_cross_entropy(start_logits, batch["start_labels"], mask=input_batch["attention_mask"]) + \
                       binary_cross_entropy(end_logits, batch["end_labels"], mask=input_batch["attention_mask"])
            else:
                loss = outputs[0]
            total_loss += loss.item()

        batch_size = batch['input_ids'].size(0)
        start_idxs, end_idxs = utils.discretize(F.softmax(outputs[1], dim=1), F.softmax(outputs[2], dim=1))
        for i in range(batch_size):
            s, e = start_idxs[i], end_idxs[i]
            answers[qids[i]] = tokenizer.decode(batch['input_ids'][i].tolist()[s:e + 1], skip_special_tokens=True)

    # Compute EM and F1 on predictions
    res = evaluate_hp.main(answers, tokenizer, adv=adv)
    return total_loss, res['exact_match'], res['f1']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="train/hotpot10_roberta", type=str)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=10, type=int)
    parser.add_argument('--learning-rate', default=3e-5, type=float)
    parser.add_argument('--weight-decay', default=0.0, type=float)
    parser.add_argument('--adam_epsilon', default=1e-8, type=float)
    parser.add_argument('--max-grad-norm', default=1.0, type=float)
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--dev-batch-size', default=16, type=int)
    parser.add_argument('--num-warmup-steps', default=0, type=int)
    parser.add_argument('--binary', action="store_true")
    parser.add_argument('--gradient-accumulation-steps', default=1, type=int)
    args = parser.parse_args()

    utils.set_random_seed(args.seed)
    config = AutoConfig.from_pretrained(BERT_MODEL, cache_dir="train/cache")
    tokenizer = RobertaTokenizerFast.from_pretrained(BERT_MODEL, cache_dir="train/cache")
    model = AutoModelForQuestionAnswering.from_pretrained(BERT_MODEL, config=config, cache_dir="train/cache")
    train_dataset = HOTPOT(tokenizer, 'new_yn_hotpot_train_v1.1.json', multi_label=args.binary)
    dev_dataset = HOTPOT(tokenizer, 'new_yn_hotpot_dev_distractor_v1.json', eval=True, multi_label=args.binary)
    adv_dev_dataset = HOTPOT(tokenizer, 'new_yn_hotpot_dev_distractor_v1_addDoc_v6.1_w_titles.json',
                             eval=True, multi_label=args.binary)

    # model = torch.load(f"{args.output_dir}/best.pt")["model"]
    # adv_dev_loss, adv_dev_em, adv_dev_f1 = test(args, model, dev_dataset, tokenizer, adv=False)
    # logger.info(f"adv EM: {adv_dev_em}, adv F1: {adv_dev_f1}, adv loss: {adv_dev_loss}")
    # exit()
    em, f1 = train(args, model, train_dataset, dev_dataset, adv_dev_dataset, tokenizer)

    with open("train/log", 'a') as f:
        f.write(f'\n{args.output_dir}, {em}, {f1}\n')


if __name__ == '__main__':
    main()
