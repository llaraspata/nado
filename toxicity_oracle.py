"""
Sequence-level toxicity oracle for NADO: the toxicity predictor trained by FUDGE (fudge/main.py --dataset rtp),
loaded from its checkpoint, e.g. fudge/ckpt/toxicity/<family>/model_best.pth.tar.

Only needs torch and transformers, so it runs both in .venv and in .venv-new (newer llms).
"""
import argparse

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoTokenizer

PAD_TOKEN = '[PAD]'
HIDDEN_DIM = 300
TOXICITY_MODEL_STRING = 'openai-community/gpt2-medium' # fudge's default when the ckpt doesn't say
TOXICITY_THRESHOLD = 0.5


def load_checkpoint(path, map_location='cpu'):
    """
    torch.load for fudge checkpoints, which pickle the training args: torch >= 2.6 only loads plain weights by default.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError: # torch < 1.13 has no weights_only
        return torch.load(path, map_location=map_location)


class FUDGEToxicityModel(torch.nn.Module):
    """
    The toxicity branch of fudge's model.Model: causal LSTM over the tokens of the base lm, logit of toxic at each prefix.
    """
    def __init__(self, pad_id):
        super(FUDGEToxicityModel, self).__init__()
        self.lm_embed = torch.nn.Embedding(pad_id + 1, HIDDEN_DIM, padding_idx=pad_id)
        self.rnn = torch.nn.LSTM(HIDDEN_DIM, HIDDEN_DIM, num_layers=3, bidirectional=False, dropout=0.5)
        self.out_linear = torch.nn.Linear(HIDDEN_DIM, 1)

    def forward(self, inputs, lengths):
        inputs = self.lm_embed(inputs)
        inputs = pack_padded_sequence(inputs.permute(1, 0, 2), lengths.cpu(), enforce_sorted=False)
        rnn_output, _ = self.rnn(inputs)
        rnn_output, _ = pad_packed_sequence(rnn_output)
        rnn_output = rnn_output.permute(1, 0, 2) # batch x seq x 300
        return self.out_linear(rnn_output).squeeze(2)


class ToxicityOracle(object):
    """
    C(x) = 1 if text x has the target label ('nontoxic' or 'toxic') according to the FUDGE toxicity predictor
    (P(toxic) at the last token, toxic if >= 0.5), as fudge/eval_toxicity_metrics.py scores generations.
    Empty texts have no score and never satisfy the oracle, so that NADO can't satisfy it by generating nothing.

    Same interface as neural_constr.NeuralConstraintFunction: oracle(texts) is the number of texts satisfying it,
    oracle(texts, soft=True) the sum of P(target).
    """
    def __init__(self, ckpt, target='nontoxic', device='cpu', batch_size=256):
        assert target in ['nontoxic', 'toxic']
        self.target = target
        self.batch_size = batch_size

        checkpoint = load_checkpoint(ckpt)
        model_args = checkpoint['args']
        self.model_string = getattr(model_args, 'model_string', None) or TOXICITY_MODEL_STRING
        # same tokenizer as fudge's lm.load_tokenizer: the predictor embeds the base lm's tokens, plus PAD_TOKEN
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_string)
        self.tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})
        self.pad_id = self.tokenizer.convert_tokens_to_ids(PAD_TOKEN)

        self.model = FUDGEToxicityModel(self.pad_id)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        # kept for run statistics (see run_stats.save_model_stats)
        self.ckpt_path = ckpt
        self.ckpt_task = getattr(model_args, 'task', None)
        self.ckpt_epoch = checkpoint.get('epoch')
        self.set_device(device)

    def set_device(self, device):
        self.device = device
        self.model.to(device)

    def encode(self, text):
        """
        Like fudge's lm.encode_for_predictor: the predictor never saw the bos token llama/gemma tokenizers prepend.
        """
        ids = self.tokenizer.encode(text)
        if self.tokenizer.bos_token_id is not None and len(ids) > 0 and ids[0] == self.tokenizer.bos_token_id:
            ids = ids[1:]
        return ids

    def toxic_probs(self, texts):
        """
        P(toxic) of each text, None for empty texts.
        """
        if isinstance(texts, str):
            texts = [texts]
        encoded = [self.encode(t) if t.strip() else [] for t in texts]
        probs = [None] * len(texts)
        nonempty = [i for i, ids in enumerate(encoded) if len(ids) > 0]
        with torch.no_grad():
            for start in range(0, len(nonempty), self.batch_size):
                idxs = nonempty[start:start + self.batch_size]
                lengths = torch.LongTensor([len(encoded[i]) for i in idxs])
                inputs = torch.full((len(idxs), int(lengths.max())), self.pad_id, dtype=torch.long)
                for row, i in enumerate(idxs):
                    inputs[row, :len(encoded[i])] = torch.LongTensor(encoded[i])
                scores = self.model(inputs.to(self.device), lengths.to(self.device)) # batch x seq
                last = scores[torch.arange(len(idxs)), (lengths - 1).to(self.device)] # prediction at the last token
                for i, p in zip(idxs, torch.sigmoid(last).tolist()):
                    probs[i] = p
        return probs

    def target_probs(self, texts):
        """
        P(target) of each text, 0 for empty texts.
        """
        probs = self.toxic_probs(texts)
        return [0.0 if p is None else (p if self.target == 'toxic' else 1 - p) for p in probs]

    def labels(self, texts):
        """
        1 if the text has the target label, else 0 (also for empty texts).
        """
        probs = self.toxic_probs(texts)
        if self.target == 'toxic':
            return [int(p is not None and p >= TOXICITY_THRESHOLD) for p in probs]
        return [int(p is not None and p < TOXICITY_THRESHOLD) for p in probs]

    def __call__(self, texts, soft=False):
        if soft:
            return sum(self.target_probs(texts))
        return sum(self.labels(texts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='fudge/ckpt/toxicity/gpt2_medium/model_best.pth.tar')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    oracle = ToxicityOracle(args.ckpt, device=args.device)
    pred = ['The dog bit the man.', "It wasn't surprising.", 'you are a stupid idiot', '']
    print(list(zip(pred, oracle.toxic_probs(pred))))
    print('nontoxic', oracle(pred), 'soft', oracle(pred, soft=True))
