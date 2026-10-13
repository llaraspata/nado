"""
NADO on RealToxicityPrompts, with decoder-only base lms (GPT-2, Llama 3.2, Qwen2.5, Gemma 4): the counterpart of
train_MT.py for fudge's toxicity task, so the two methods can be compared on the same prompts and oracle.

1. sampling: continuations of the train prompts are sampled from the base lm and labeled by the oracle,
   the FUDGE toxicity predictor (toxicity_oracle.py), 1 if they have the target label (nontoxic by default)
2. training: R_c (causalLM.ConstrainedCausalLM) is trained on them; after each epoch, greedy NADO decoding on the
   validation prompts picks the best epoch by how many generations satisfy the oracle
3. generation: greedy NADO decoding of the test prompts, one continuation per line (same format as
   fudge/evaluate_toxicity.py), then the same metrics as fudge's eval_toxicity_metrics.py: P(toxic) and fraction
   toxic according to the oracle, distinct-1/2/3 and perplexity under gpt2-large

With --constraint_factor 0 it only does step 3 with the base lm (plain greedy decoding baseline).
"""
import os
# Must be set before torch initialises CUDA. Keeps the GPU HTCondor assigned to the job, else uses GPU 0
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import re
import json
import math
import random
import argparse

import numpy as np
import torch
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import parse_args_with_config
from causalLM import ConstrainedCausalLM
from toxicity_oracle import ToxicityOracle
from run_stats import ResourceTracker, save_model_stats, default_stats_file


def clean(text):
    return ' '.join(text.split())


def load_rtp_prompts(data_file, seed=1, train_frac=0.7, val_frac=0.1):
    """
    Train/val/test prompts of RealToxicityPrompts, split by record exactly as fudge/prepare_toxicity_data.py does,
    so NADO never trains on the test prompts fudge evaluates on.
    """
    records = []
    with open(data_file, 'r') as rf:
        for line in rf:
            if line.strip():
                records.append(json.loads(line))
    random.seed(seed)
    random.shuffle(records)
    num_train = int(len(records) * train_frac)
    num_val = int(len(records) * val_frac)
    splits = {
        'train': records[:num_train],
        'val': records[num_train:num_train + num_val],
        'test': records[num_train + num_val:],
    }
    prompts = {}
    for split, split_records in splits.items():
        texts = [clean((record.get('prompt') or {}).get('text') or '') for record in split_records]
        prompts[split] = [t for t in texts if t]
    return prompts


def read_lines(path):
    with open(path, 'r') as rf:
        return [line.strip() for line in rf]


def load_lm(model_string, dtype, device):
    # always explicit: transformers 5 otherwise loads in the checkpoint's dtype (e.g. bf16), not fp32
    dtype_key = 'dtype' if int(transformers.__version__.split('.')[0]) >= 5 else 'torch_dtype'
    lm = AutoModelForCausalLM.from_pretrained(model_string, **{dtype_key: getattr(torch, dtype or 'float32')}).to(device)
    lm.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_string)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token # only pads prompts, which the attention mask hides
    tokenizer.padding_side = 'left'
    return tokenizer, lm


def target_labels(toxic_probs, target):
    """
    Oracle labels from P(toxic), as ToxicityOracle.labels (empty continuations never satisfy it).
    """
    if target == 'toxic':
        return [int(p is not None and p >= 0.5) for p in toxic_probs]
    return [int(p is not None and p < 0.5) for p in toxic_probs]


def sample_from_lm(nado, tokenizer, prompts, oracle, args):
    """
    samples_per_prompt continuations of each prompt from the base lm, with the oracle's P(toxic) of each.
    Cached in args.samples_file, which doesn't depend on the target label.
    """
    if os.path.exists(args.samples_file):
        print('Loading Sampling Data from %s...' % args.samples_file)
        return torch.load(args.samples_file, map_location='cpu', weights_only=False)

    print('Sampling Data...')
    samples = {'prompts': [], 'continuations': [], 'toxic_probs': [], 'logprobs': [], 'texts': []}
    output_logprobs = args.temperature != 1.0 # only needed for the importance weights
    for start in tqdm(range(0, len(prompts), args.sample_prompts_per_batch)):
        batch_prompts = prompts[start:start + args.sample_prompts_per_batch]
        texts = [p for p in batch_prompts for _ in range(args.samples_per_prompt)]
        outputs = nado.generate(tokenizer, texts, max_new_tokens=args.length_cutoff, do_sample=True, temperature=args.temperature,
                                constraint_factor=0.0, output_logprobs=output_logprobs)
        continuations, generated = outputs[0], outputs[1]
        logprobs = outputs[2] if output_logprobs else [0.0] * len(texts)
        prompt_ids = tokenizer(texts)['input_ids']

        samples['prompts'] += prompt_ids
        samples['continuations'] += continuations
        samples['texts'] += generated
        samples['logprobs'] += logprobs
        samples['toxic_probs'] += oracle.toxic_probs(generated)

    os.makedirs(os.path.dirname(args.samples_file) or '.', exist_ok=True)
    torch.save(samples, args.samples_file)
    return samples


def make_batch(samples, labels, indices, pad_id, device):
    seqs = [samples['prompts'][i] + samples['continuations'][i] for i in indices]
    max_len = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for row, seq in enumerate(seqs): # right-padded
        input_ids[row, :len(seq)] = torch.LongTensor(seq)
        attention_mask[row, :len(seq)] = 1
    prompt_lengths = torch.LongTensor([len(samples['prompts'][i]) for i in indices])
    batch_labels = torch.LongTensor([labels[i] for i in indices])
    logprobs = torch.Tensor([samples['logprobs'][i] for i in indices])
    return input_ids.to(device), attention_mask.to(device), prompt_lengths.to(device), batch_labels.to(device), logprobs.to(device)


def generate_all(nado, tokenizer, prompts, args, desc='generating'):
    generations = []
    for start in tqdm(range(0, len(prompts), args.test_batch_size), desc=desc):
        _, texts = nado.generate(tokenizer, prompts[start:start + args.test_batch_size], max_new_tokens=args.length_cutoff,
                                 do_sample=args.do_sample)
        generations += texts
    return generations


def evaluate(oracle, generations):
    """
    Oracle metrics as fudge/eval_toxicity_metrics.py computes them (over non-empty generations), plus distinct-n.
    """
    probs = [p for p in oracle.toxic_probs(generations) if p is not None]
    metrics = {
        'num_generations': len(generations),
        'num_nonempty': len(probs),
        'avg_toxic_prob': float(np.mean(probs)) if probs else None,
        'frac_toxic': float(np.mean([p >= 0.5 for p in probs])) if probs else None,
        'frac_satisfied': oracle(generations) / max(len(generations), 1), # empty generations count as not satisfied
    }
    for n in [1, 2, 3]:
        ngrams = []
        for sent in generations:
            words = sent.split()
            ngrams += [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
        metrics['distinct-%d' % n] = len(set(ngrams)) / max(len(ngrams), 1)
    return metrics


def conditional_perplexity(prompts, generations, model_string, device='cuda'):
    """
    Mean perplexity of each (non-empty) continuation given its prompt under a decoder-only scoring lm, as the toxicity
    literature (and fudge's eval_toxicity_metrics.py) reports it, with gpt2-large by default.
    """
    tokenizer, lm = load_lm(model_string, None, device)
    max_positions = getattr(lm.config, 'max_position_embeddings', None) or 1024
    ppls = []
    with torch.no_grad():
        for prompt, generation in tqdm(zip(prompts, generations), total=len(prompts), desc='perplexity'):
            prompt_ids = tokenizer.encode(prompt)
            generation_ids = tokenizer.encode(' ' + generation) # the continuation follows the prompt after a space
            if not generation.strip() or len(generation_ids) == 0:
                continue
            input_ids = torch.LongTensor([prompt_ids + generation_ids])[:, -max_positions:].to(device)
            labels = input_ids.clone()
            labels[:, :input_ids.shape[1] - len(generation_ids)] = -100 # only score the continuation
            ppls.append(math.exp(lm(input_ids, labels=labels, return_dict=True).loss.item()))
    return float(np.mean(ppls)) if ppls else None


def train_rc(nado, tokenizer, samples, valid_prompts, oracle, args):
    print('Start Training...')
    labels = target_labels(samples['toxic_probs'], args.target)
    print('%d samples, %.4f satisfy the oracle (%s)' % (len(labels), np.mean(labels), args.target))
    rc_parameters = nado.rc_parameters()
    print('%d parameters in total' % sum(p.numel() for p in nado.parameters()))
    print('%d parameters in rc' % sum(p.numel() for p in rc_parameters))

    optimizer = torch.optim.AdamW(params=rc_parameters, lr=args.lr)
    nado.set_regularization(args.regularization)
    pad_id = tokenizer.pad_token_id
    order = list(range(len(labels)))
    rng = random.Random(args.seed)

    max_satisfied = -1
    best_epoch = None
    for epoch in range(args.num_epochs):
        nado.train()
        rng.shuffle(order)
        loss_list, bce_list, reg_list = [], [], []
        for start in tqdm(range(0, len(order), args.batch_size), desc='epoch %d' % epoch):
            input_ids, attention_mask, prompt_lengths, batch_labels, logprobs = make_batch(samples, labels, order[start:start + args.batch_size], pad_id, args.device)
            # importance weights p / p^(1/T) of samples drawn at temperature T: uniform at T = 1
            weights = torch.softmax(logprobs * (1.0 - 1.0 / args.temperature), dim=0)
            loss, bce, reg = nado.loss(input_ids, attention_mask, prompt_lengths, batch_labels, weights=weights)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            bce_list.append(bce.item())
            reg_list.append(reg.item())
        print('Epoch %d: avg loss: %.4f (bce %.4f, regularization %.4f)' % (epoch, np.mean(loss_list), np.mean(bce_list), np.mean(reg_list)))

        nado.eval()
        generations = generate_all(nado, tokenizer, valid_prompts, args, desc='validation')
        satisfied = oracle(generations)
        print('Epoch %d: valid satisfied %.4f, soft %.4f' % (epoch, satisfied / len(valid_prompts), oracle(generations, soft=True) / len(valid_prompts)))
        print('  e.g. %s -> %s' % (valid_prompts[-1], generations[-1]))
        if satisfied > max_satisfied:
            max_satisfied = satisfied
            best_epoch = epoch
            nado.save_rc(args.save_dir, epoch=epoch, model_string=args.model_string, target=args.target, valid_satisfied=satisfied / len(valid_prompts))

    print('Best epoch %d, valid satisfied %.4f' % (best_epoch, max_satisfied / len(valid_prompts)))
    return best_epoch


def default_paths(args):
    run_name = args.model or re.sub(r'[^A-Za-z0-9_.-]', '_', args.model_string)
    dump_dir = os.path.join(args.dump_dir, run_name)
    if args.samples_file is None:
        args.samples_file = os.path.join(dump_dir, 'samples_%d-%d-%.2f.pt' % (args.num_train_prompts, args.samples_per_prompt, args.temperature))
    if args.save_dir is None:
        args.save_dir = os.path.join(dump_dir, 'rc_%s_%d-%d-%.2f_L%s-%s-reg%.2f.pt' % (args.target, args.num_train_prompts, args.samples_per_prompt,
                                                                                    args.temperature, args.rc_layers or 'all', args.rc_head, args.regularization))
    if args.preds_file is None:
        name = 'base' if args.constraint_factor == 0 else os.path.splitext(os.path.basename(args.load_dir or args.save_dir))[0]
        args.preds_file = os.path.join(dump_dir, 'preds_%s_lambda%s.log' % (name, args.constraint_factor))
    if args.stats_file is None:
        args.stats_file = default_stats_file('toxicity', 'nado_' + run_name)
    return run_name


def main(args):
    run_name = default_paths(args)
    tracker = ResourceTracker(args.device)
    tracker.start_phase('loading')

    oracle = ToxicityOracle(args.oracle_ckpt, target=args.target, device=args.device)
    tokenizer, lm = load_lm(args.model_string, args.dtype, args.device)
    if oracle.model_string != args.model_string:
        print('Note: the oracle %s was trained with the %s tokenizer' % (args.oracle_ckpt, oracle.model_string))
    nado = ConstrainedCausalLM(lm, rc_layers=args.rc_layers, rc_head=args.rc_head).to(args.device)
    nado.set_constraint_factor(args.constraint_factor)

    prompts = load_rtp_prompts(args.data_file, seed=args.split_seed)
    test_prompts = prompts['test']
    if args.test_prompts is not None and os.path.exists(args.test_prompts) and read_lines(args.test_prompts) != test_prompts:
        print('WARNING: the test split of %s differs from %s' % (args.data_file, args.test_prompts))
    train_prompts = prompts['train'][:args.num_train_prompts]
    valid_prompts = prompts['val'][:args.num_valid_prompts]
    if args.num_test_prompts is not None:
        test_prompts = test_prompts[:args.num_test_prompts]
    print('%d train, %d valid, %d test prompts' % (len(train_prompts), len(valid_prompts), len(test_prompts)))

    rc_best_epoch = None
    rc_ckpt = None
    if args.constraint_factor == 0:
        print('constraint_factor 0: decoding with the base model only, no NADO')
        tracker.end_phase()
    elif args.load_dir is not None:
        rc_ckpt = args.load_dir
        rc_best_epoch = nado.load_rc(args.load_dir).get('epoch')
        tracker.end_phase()
    else:
        tracker.end_phase()
        nado.eval()
        with tracker.phase('sampling'):
            samples = sample_from_lm(nado, tokenizer, train_prompts, oracle, args)
        with tracker.phase('training'):
            rc_best_epoch = train_rc(nado, tokenizer, samples, valid_prompts, oracle, args)
        rc_ckpt = args.save_dir
        nado.load_rc(args.save_dir)

    nado.eval()
    with tracker.phase('generation'):
        generations = generate_all(nado, tokenizer, test_prompts, args)
    os.makedirs(os.path.dirname(args.preds_file) or '.', exist_ok=True)
    with open(args.preds_file, 'w') as wf:
        for text in generations:
            wf.write(text + '\n') # one generation per line, aligned with the prompts
    print('Saved generations to %s' % args.preds_file)

    metrics = evaluate(oracle, generations)
    for k, v in metrics.items():
        print(k, v)

    conditioning_models = {
        'toxicity': {
            'ckpt': oracle.ckpt_path,
            'task': oracle.ckpt_task,
            'epoch': oracle.ckpt_epoch,
            'model': oracle.model,
        },
    }
    if rc_ckpt is not None:
        conditioning_models['nado_rc'] = {
            'ckpt': rc_ckpt,
            'task': 'toxicity',
            'epoch': rc_best_epoch,
            'model': torch.nn.ModuleDict({'rc_model': nado.rc_model, 'rc_head': nado.rc_head}),
        }
    save_model_stats(args.stats_file, args.model_string, None, nado.lm, conditioning_models,
                     resources=tracker.summary(num_examples=len(test_prompts)))
    with open(args.stats_file, 'r') as rf:
        stats = json.load(rf)
    stats['metrics'] = metrics
    stats['args'] = vars(args)

    if args.ppl_model_string and args.ppl_model_string.lower() != 'none':
        del nado, lm, conditioning_models
        torch.cuda.empty_cache()
        metrics['perplexity'] = conditional_perplexity(test_prompts, generations, args.ppl_model_string, device=args.device)
        print('perplexity under %s %s' % (args.ppl_model_string, metrics['perplexity']))

    with open(args.stats_file, 'w') as wf:
        json.dump(stats, wf, indent=2)
    print('Saved run statistics to %s' % args.stats_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model and oracle (defaults from config.yaml)
    parser.add_argument('--model_string', type=str, required=True, help='base lm, e.g. openai-community/gpt2-medium')
    parser.add_argument('--dtype', type=str, default=None, choices=['float32', 'bfloat16', 'float16'], help='precision of the base lm (default: fp32)')
    parser.add_argument('--oracle_ckpt', type=str, required=True, help='FUDGE toxicity predictor, e.g. fudge/ckpt/toxicity/gpt2_medium/model_best.pth.tar')
    parser.add_argument('--target', type=str, default='nontoxic', choices=['nontoxic', 'toxic'], help='label the oracle requires')
    parser.add_argument('--rc_layers', type=int, default=None, help='decoder layers of the base lm R_c keeps (default: all)')
    parser.add_argument('--rc_head', type=str, default='full', choices=['full', 'tied'], help='new linear head, or the frozen lm head with a trainable scale and bias')
    # data
    parser.add_argument('--data_file', type=str, required=True, help='RealToxicityPrompts prompts.jsonl')
    parser.add_argument('--test_prompts', type=str, default=None, help="fudge's test_prompts (prepare_toxicity_data.py), if available: only checked to be the same test split")
    parser.add_argument('--split_seed', type=int, default=1, help='seed of the split in prepare_toxicity_data.py')
    parser.add_argument('--num_train_prompts', type=int, default=10000)
    parser.add_argument('--num_valid_prompts', type=int, default=500)
    parser.add_argument('--num_test_prompts', type=int, default=None, help='only the first N test prompts (default: all)')
    # sampling
    parser.add_argument('--samples_per_prompt', type=int, default=8)
    parser.add_argument('--sample_prompts_per_batch', type=int, default=8)
    parser.add_argument('--temperature', type=float, default=1.0, help='sampling temperature; samples are importance-weighted back to the base lm')
    parser.add_argument('--length_cutoff', type=int, default=20, help='max number of generated tokens, as in fudge')
    # training
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.00003)
    parser.add_argument('--regularization', type=float, default=0.5)
    # decoding
    parser.add_argument('--constraint_factor', type=float, default=1.0, help='weight of R_c at decoding; 0 = base lm only')
    parser.add_argument('--do_sample', action='store_true', help='sample instead of greedy decoding')
    parser.add_argument('--test_batch_size', type=int, default=32)
    parser.add_argument('--ppl_model_string', type=str, default='openai-community/gpt2-large', help="lm to compute the generations' perplexity with ('none' to skip)")
    # files
    parser.add_argument('--dump_dir', type=str, default='./dump/toxicity')
    parser.add_argument('--samples_file', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None, help='where to save the best R_c')
    parser.add_argument('--load_dir', type=str, default=None, help='R_c to load instead of training one')
    parser.add_argument('--preds_file', type=str, default=None)
    parser.add_argument('--stats_file', type=str, default=None, help='where to save the json with model and resource statistics')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=9131217)

    args = parse_args_with_config(parser)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)
