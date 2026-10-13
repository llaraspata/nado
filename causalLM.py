"""
NADO for decoder-only language models (GPT-2, Llama 3.2, Qwen2.5, Gemma 4), the counterpart of marianMT.ConstrainedIndMT.

The base lm p is frozen. R_c is a copy of its text backbone plus a head giving, at each position, the logit of
R(y_<=i) = P(oracle satisfied | prefix y_<=i) for every candidate next token. Decoding samples from
    q(y_i | y_<i) ∝ p(y_i | y_<i) * R(y_<=i) ^ constraint_factor.

To fit the newer llms on one gpu, R_c can keep only the first rc_layers decoder layers of the base model, and use the
'tied' head (the frozen lm head of the base model, with a trainable scale and bias) instead of a new hidden x vocab
linear layer. Its embeddings are always shared with the base model and frozen.

Only needs torch and transformers (4.x in .venv, 5.x in fudge/.venv-new).
"""
import copy
import contextlib

import torch
import torch.nn.functional as F
from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList


def text_backbone(lm):
    """
    The transformer under the lm head: e.g. GPT2Model, LlamaModel, Qwen2Model, and the text model of multimodal Gemma 4.
    """
    base = lm.base_model
    return getattr(base, 'language_model', base)


def decoder_layers_name(backbone):
    for name in ['layers', 'h']: # llama/qwen/gemma, gpt2
        if isinstance(getattr(backbone, name, None), torch.nn.ModuleList):
            return name
    raise ValueError('cannot find the decoder layers of {}'.format(type(backbone).__name__))


class TiedRcHead(torch.nn.Module):
    """
    R_c head reusing the frozen lm head weight W: logits = scale * W h + bias. scale and bias start at 0, so R = 0.5
    everywhere at first, as with the zero-initialized linear head.
    """
    def __init__(self, vocab_size):
        super(TiedRcHead, self).__init__()
        self.scale = torch.nn.Parameter(torch.zeros(1))
        self.bias = torch.nn.Parameter(torch.zeros(vocab_size))

    def forward(self, hidden, weight):
        return F.linear(hidden.to(weight.dtype), weight).float() * self.scale + self.bias


class ConstrainedCausalLM(torch.nn.Module):
    def __init__(self, lm, rc_layers=None, rc_head='full'):
        super(ConstrainedCausalLM, self).__init__()
        self.lm = lm
        self.lm.eval()
        for p in self.lm.parameters():
            p.requires_grad = False

        backbone = text_backbone(lm)
        layers_name = decoder_layers_name(backbone)
        layers = getattr(backbone, layers_name)
        self.rc_layers = len(layers) if rc_layers is None else rc_layers
        assert 1 <= self.rc_layers <= len(layers), 'rc_layers must be between 1 and {}'.format(len(layers))

        # copy the backbone, sharing (not copying) the embeddings and the layers R_c drops
        shared = [m for m in backbone.modules() if isinstance(m, torch.nn.Embedding)] + list(layers[self.rc_layers:])
        self.rc_model = copy.deepcopy(backbone, {id(m): m for m in shared})
        # the forward passes of these models only run the layers in the list, whatever the config says
        setattr(self.rc_model, layers_name, torch.nn.ModuleList(list(getattr(self.rc_model, layers_name))[:self.rc_layers]))
        shared_params = set(id(p) for m in shared for p in m.parameters())
        for p in self.rc_model.parameters():
            if id(p) not in shared_params:
                p.data = p.data.float() # trained in fp32: bf16 weights would swallow most updates
                p.requires_grad = True

        vocab_size, hidden_size = lm.get_output_embeddings().weight.shape
        self.rc_head_type = rc_head
        if rc_head == 'full':
            self.rc_head = torch.nn.Linear(hidden_size, vocab_size, bias=True)
            torch.nn.init.zeros_(self.rc_head.weight)
            torch.nn.init.zeros_(self.rc_head.bias)
        elif rc_head == 'tied':
            self.rc_head = TiedRcHead(vocab_size)
        else:
            raise ValueError('unknown rc_head {}'.format(rc_head))

        self.constraint_factor = 1.0
        self.regularization = 0.5
        # the frozen parts of R_c keep the lm's dtype (e.g. bf16), so run it under autocast
        self.autocast_dtype = lm.dtype if lm.dtype in (torch.bfloat16, torch.float16) else None

    def train(self, mode=True):
        super(ConstrainedCausalLM, self).train(mode)
        self.lm.eval() # the base model stays frozen, without dropout
        return self

    def set_constraint_factor(self, constraint_factor):
        self.constraint_factor = constraint_factor

    def set_regularization(self, regularization):
        self.regularization = regularization

    def rc_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def rc_forward(self, input_ids, attention_mask, position_ids=None, past_key_values=None, use_cache=False):
        """
        Returns (logits of R(y_<=j, v) for every position j and next token v, cache).
        """
        autocast = torch.autocast(input_ids.device.type, dtype=self.autocast_dtype) if self.autocast_dtype is not None else contextlib.nullcontext()
        with autocast:
            outputs = self.rc_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                                    past_key_values=past_key_values, use_cache=use_cache, return_dict=True)
        return outputs.last_hidden_state, outputs.past_key_values

    def rc_logits(self, hidden):
        if self.rc_head_type == 'tied':
            return self.rc_head(hidden, self.lm.get_output_embeddings().weight)
        return self.rc_head(hidden.float())

    def loss(self, input_ids, attention_mask, prompt_lengths, labels, weights=None):
        """
        NADO loss on sampled sequences, as ConstrainedIndMT.forward with rc_labels (reg_type 3).
        input_ids: batch x seq, prompt + sampled continuation, right-padded; attention_mask: batch x seq
        prompt_lengths: batch; labels: batch, oracle label of each continuation
        weights: batch, importance weights summing to 1 (default: uniform)
        Returns (loss, bce part, regularization part).
        """
        batch_size, seq_len = input_ids.shape
        lengths = attention_mask.sum(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        # positions j whose next token y_j+1 is a continuation token: R(y_<=j+1) is predicted at j
        pred_mask = (positions >= prompt_lengths.unsqueeze(1) - 1) & (positions < lengths.unsqueeze(1) - 1)
        if weights is None:
            weights = torch.full((batch_size,), 1.0 / batch_size, device=input_ids.device)
        batch_index = torch.arange(batch_size, device=input_ids.device).unsqueeze(1).expand(-1, seq_len)[pred_mask]

        hidden, _ = self.rc_forward(input_ids, attention_mask)
        constr_logits = self.rc_logits(hidden[pred_mask]) # N x vocab, R(y_<=j, v) at each prediction position
        next_ids = torch.cat([input_ids[:, 1:], input_ids[:, :1]], dim=1)[pred_mask]
        pred_logits = constr_logits.gather(1, next_ids.unsqueeze(1)).squeeze(1) # N, R(y_<=j+1)

        bce = F.binary_cross_entropy_with_logits(pred_logits, labels.float()[batch_index], reduction='none')
        loss1 = (weights[batch_index] * bce).sum()
        loss2 = torch.zeros_like(loss1)

        if self.regularization > 0.0:
            # R(y_<=j) = sum_v p(v | y_<=j) R(y_<=j, v), for continuation tokens y_j that have a next token
            with torch.no_grad():
                lm_logits = self.lm(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).logits
                lm_logprobs = F.log_softmax(lm_logits[pred_mask].float(), dim=-1)
                del lm_logits
            # entries are (b, j) in row-major order, so the entry before (b, j) is (b, j-1), except at each row's first j
            first = (positions == prompt_lengths.unsqueeze(1) - 1)[pred_mask]
            reg_index = (~first).nonzero().squeeze(1)
            # keep sum_probs < 1 so sum_logits stays finite
            sum_log = torch.logsumexp(lm_logprobs[reg_index] + F.logsigmoid(constr_logits[reg_index]), dim=-1).clamp(max=-1e-6)
            sum_logits = sum_log - torch.log1p(-torch.exp(sum_log))
            prev_logits = pred_logits[reg_index - 1] # R(y_<=j)
            prev_probs = torch.sigmoid(prev_logits)
            # KL between the two Bernoullis, averaged over the batch
            kl = F.binary_cross_entropy_with_logits(sum_logits, prev_probs, reduction='none') \
                - F.binary_cross_entropy_with_logits(prev_logits, prev_probs, reduction='none')
            loss2 = self.regularization * kl.sum() / batch_size

        return loss1 + loss2, loss1.detach(), loss2.detach()

    @torch.no_grad()
    def generate(self, tokenizer, prompts, max_new_tokens=20, do_sample=False, temperature=1.0, constraint_factor=None, output_logprobs=False):
        """
        Continue each prompt. tokenizer must pad on the left. With constraint_factor 0 (or R_c not used), plain decoding
        from the base model. Sampling is ancestral sampling at the given temperature (no top-k / top-p).
        Returns (continuation token ids, ending with eos if generated; texts; and, with output_logprobs, the log prob
        of each continuation under the base model).
        """
        factor = self.constraint_factor if constraint_factor is None else constraint_factor
        device = self.lm.device
        encoded = tokenizer(prompts, return_tensors='pt', padding=True).to(device)
        processors = LogitsProcessorList([VocabLimitLogitsProcessor(len(tokenizer))])
        if factor != 0:
            processors.append(NADOLogitsProcessor(self, encoded['attention_mask'], factor))

        kwargs = {'do_sample': do_sample}
        if do_sample:
            kwargs.update(temperature=temperature, top_k=0, top_p=1.0)
        outputs = self.lm.generate(**encoded,
                                   generation_config=self.generation_config(tokenizer),
                                   max_new_tokens=max_new_tokens,
                                   logits_processor=processors,
                                   return_dict_in_generate=True,
                                   output_logits=output_logprobs,
                                   **kwargs)

        eos_ids = self.eos_token_ids(tokenizer)
        generated = outputs.sequences[:, encoded['input_ids'].shape[1]:].tolist()
        continuations = []
        for ids in generated:
            end = next((i + 1 for i, t in enumerate(ids) if t in eos_ids), len(ids)) # keep the eos, drop the padding after it
            continuations.append(ids[:end])
        texts = [' '.join(tokenizer.decode(ids, skip_special_tokens=True).split()) for ids in continuations]
        if not output_logprobs:
            return continuations, texts

        logprobs = []
        step_logprobs = [F.log_softmax(l.float(), dim=-1) for l in outputs.logits] # raw lm logits, before the processors
        for b, ids in enumerate(continuations):
            logprobs.append(sum(step_logprobs[t][b, token].item() for t, token in enumerate(ids)))
        return continuations, texts, logprobs

    def eos_token_ids(self, tokenizer):
        eos = self.lm.generation_config.eos_token_id
        if eos is None:
            eos = tokenizer.eos_token_id
        return set(eos if isinstance(eos, list) else [eos])

    def generation_config(self, tokenizer):
        """
        A clean generation config: only the special tokens of the model's own, so its default sampling settings
        (e.g. llama's temperature and top-p) don't leak in.
        """
        defaults = self.lm.generation_config
        eos = defaults.eos_token_id if defaults.eos_token_id is not None else tokenizer.eos_token_id
        return GenerationConfig(bos_token_id=defaults.bos_token_id, eos_token_id=eos, pad_token_id=tokenizer.pad_token_id)

    def rc_state_dict(self):
        return {name: p.detach().cpu() for name, p in self.named_parameters() if p.requires_grad}

    def save_rc(self, path, **info):
        print('Saving R_c to %s...' % path)
        torch.save({'state_dict': self.rc_state_dict(), 'rc_layers': self.rc_layers, 'rc_head': self.rc_head_type, **info}, path)

    def load_rc(self, path):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        assert checkpoint['rc_layers'] == self.rc_layers and checkpoint['rc_head'] == self.rc_head_type, \
            'R_c in {} has rc_layers={} rc_head={}'.format(path, checkpoint['rc_layers'], checkpoint['rc_head'])
        missing, unexpected = self.load_state_dict(checkpoint['state_dict'], strict=False)
        assert not unexpected, unexpected
        trainable = set(name for name, p in self.named_parameters() if p.requires_grad)
        missing = [name for name in missing if name in trainable]
        assert not missing, missing
        return checkpoint


class VocabLimitLogitsProcessor(LogitsProcessor):
    """
    Never generate ids from vocab_size on (e.g. qwen's unused embedding rows), as fudge's decoding does.
    """
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def __call__(self, input_ids, scores):
        if scores.shape[1] > self.vocab_size:
            scores[:, self.vocab_size:] = -float('inf')
        return scores


class NADOLogitsProcessor(LogitsProcessor):
    """
    Turns the base lm's next token scores into log p(v | y_<i) + factor * log R(y_<i, v), running R_c alongside
    with its own kv cache. attention_mask is the (left-padded) prompts' mask.
    """
    def __init__(self, nado, attention_mask, factor):
        self.nado = nado
        self.attention_mask = attention_mask
        self.factor = factor
        self.past = None
        self.last_positions = None

    def __call__(self, input_ids, scores):
        if self.past is None: # first step: the whole prompt
            new_ids = input_ids
            position_ids = (self.attention_mask.cumsum(-1) - 1).clamp(min=0)
        else: # then the token chosen at the previous step
            new_ids = input_ids[:, -1:]
            self.attention_mask = torch.cat([self.attention_mask, self.attention_mask.new_ones((input_ids.shape[0], 1))], dim=1)
            position_ids = self.last_positions + 1
        hidden, self.past = self.nado.rc_forward(new_ids, self.attention_mask, position_ids=position_ids, past_key_values=self.past, use_cache=True)
        self.last_positions = position_ids[:, -1:]
        constr_logits = self.nado.rc_logits(hidden[:, -1])
        return F.log_softmax(scores.float(), dim=-1) + self.factor * F.logsigmoid(constr_logits)
