from argparse import ArgumentParser
import numpy as np
import os
import string
from time import time
import torch
from torch.nn import functional as F
from torch.optim import Adam
from tqdm import tqdm
from wurlitzer import pipes

from .executor import Executor
from .generator import Generator


def parse_flags():
    a = ArgumentParser()
    a.add_argument('--device', type=str, default='cuda:0')
    a.add_argument('--num_epochs', type=int, default=1000)
    a.add_argument('--batches_per_epoch', type=int, default=100)
    a.add_argument('--batch_size', type=int, default=32)
    a.add_argument('--z_dim', type=int, default=128)
    a.add_argument('--g_width', type=int, default=32)
    a.add_argument('--e_width', type=int, default=32)
    a.add_argument('--checkpoint', type=str, default='data/checkpoint.pt')
    a.add_argument('--log', type=str, default='data/log.txt')
    return a.parse_args()


def alphabet_to_mappings(alphabet, unk_chr):
    x = []
    for c in alphabet:
        n = ord(c)
        x.append(n)
    x = np.array(x, dtype=np.int64)
    assert (x < 256).all()
    assert len(set(x)) == len(x)
    decode = torch.from_numpy(x)

    unk_id = alphabet.index(unk_chr)
    assert unk_id != -1
    encode = torch.full((256,), unk_id, dtype=torch.int64)
    encode[decode] = torch.arange(len(decode), dtype=torch.int64)

    return encode, decode


def texts_to_tensor(encode, texts):
    nnn = []
    for s in texts:
        nn = list(map(ord, s))
        nnn.append(nn)
    x = np.array(nnn, np.int64)
    assert (x < 256).all()
    x = torch.from_numpy(x)
    return encode[x]


def tensor_to_texts(decode, x):
    x = decode[x]
    x = x.to(torch.uint8)
    nnn = x.detach().cpu().numpy()
    ss = []
    for nn in nnn:
        s = nn.tobytes().decode('utf-8')
        ss.append(s)
    return ss


class Dictionary(object):
    def __init__(self, alphabet, unk_chr, device):
        self.alphabet = alphabet
        self.unk_chr = unk_chr
        self.device = device

        encode, decode = alphabet_to_mappings(alphabet, unk_chr)
        self.encode = encode.to(device)
        self.decode = decode.to(device)

    def texts_to_tensor(self, ss):
        return texts_to_tensor(self.encode, ss)

    def tensor_to_texts(self, x):
        return tensor_to_texts(self.decode, x)


def python(s):
    s = 'print(%s)' % s
    if '**' in s:
        r = 'TooBig'
        ok = False
    else:
        try:
            with pipes() as (out, err):
                exec(s)
            r = out.read()
            r = r[:-1]
            ok = True
        except SyntaxError as e:
            #r = '%d:%d:%s:%s' % (e.lineno, e.offset, e.text, str(e))
            #r = '%d :: %d :: %s' % (e.lineno, e.offset, str(e))
            #r = 'SyntaxError %d:%d' % (e.lineno, e.offset)
            r = e.__class__.__name__
            ok = False
        except Exception as e:
            #r = str(e)
            #r = 'other'
            r = e.__class__.__name__
            ok = False
    return r, ok


class Python(object):
    def __init__(self, g_dict, e_dict, e_width):
        self.g_dict = g_dict
        self.e_dict = e_dict
        self.e_width = e_width

    def __call__(self, x):
        indices = x.max(1)[1]
        ss = self.g_dict.tensor_to_texts(indices)
        rr = []
        oks = []
        for s in ss:
            r, ok = python(s)
            r = r.ljust(self.e_width)[:self.e_width]
            rr.append(r)
            oks.append(ok)
        r = self.e_dict.texts_to_tensor(rr)
        in_text = ss
        out_text = rr
        return r, in_text, out_text, oks


def pick_widths(batch_size, max_width):
    unscaled = 1 + np.random.random(batch_size) * 9
    frac = 1 - np.log(unscaled) / np.log(10)
    x = 1 + (max_width - 1) * frac
    return x.astype(np.int64)


def widths_to_mask(ww, max_width, device):
    xx = []
    for w in ww:
        x = [0] * w + [1] * (max_width - w)
        xx.append(x)
    return torch.tensor(xx, dtype=torch.float32, device=device)


def stats(x):
    mean = np.mean(x)
    std = np.std(x)
    bar = '%s%s' % ('=' * int(100 * mean), '-' * int(100 * std))
    return '%.3f+%.3f %s' % (mean, std, bar)


def render_epoch_results(e, b):
    indent = ' ' * 4
    lines = [
        'Epoch %d' % e['epoch'],
    ]

    head = '\n'.join(lines)

    lines = [
        '',
        '%sLoss' % indent,
        '%s----' % indent,
        '',
        'Gen Chr:  %s' % stats(e['gen_chr_loss']),
        'Gen Nul:  %s' % stats(e['gen_nul_loss']),
        'Gen Std:  %s' % stats(e['gen_std_loss']),
        'Executor: %s' % stats(e['exe_loss']),
        'Noncrash: %s' % stats(e['ok_loss']),
        '',
        '%sBatch' % indent,
        '%s-----' % indent,
        '',
    ]

    code_len = len(b['in_text'][0])
    pred_out_len = len(b['pred_out_text'][0])
    true_out_len = len(b['true_out_text'][0])

    line = 'Width %s %s %s' % \
        ('Code'.ljust(code_len), 'Predicted'.ljust(pred_out_len),
         'Actual'.ljust(true_out_len))
    lines.append(line)

    bars = '----- %s %s %s' % ('-' * code_len, '-' * pred_out_len,
                               '-' * true_out_len)
    lines.append(bars)

    batch_size = len(b['widths'])
    for i in range(batch_size):
        width = b['widths'][i]
        code = b['in_text'][i]
        pred_out = b['pred_out_text'][i]
        true_out = b['true_out_text'][i]
        line = '%5d %s %s %s' % (width, code, pred_out, true_out)
        lines.append(line)

    bars = '----- %s %s %s' % ('-' * code_len, '-' * pred_out_len,
                               '-' * true_out_len)
    lines.append(bars)

    lines.append('')

    lines = list(map(lambda s: '%s%s' % (indent, s) if s else '', lines))

    body = '\n'.join(lines)

    return '\n'.join([head, body])


def main(flags):
    device = torch.device(flags.device)

    g_alphabet = ' ' + string.digits + '+-*=>()'
    g_unk = ' '
    g_dict = Dictionary(g_alphabet, g_unk, device)

    e_alphabet = list(map(chr, range(32, 127)))
    e_unk = ' '
    e_dict = Dictionary(e_alphabet, e_unk, device)

    python = Python(g_dict, e_dict, flags.e_width)

    generator = Generator(flags.z_dim, len(g_alphabet), flags.g_width)
    generator.to(device)

    executor = Executor(len(g_alphabet), flags.g_width, len(e_alphabet),
                        flags.e_width)
    executor.to(device)

    g_optimizer = Adam(generator.parameters())
    e_optimizer = Adam(executor.parameters())

    if flags.checkpoint and os.path.exists(flags.checkpoint):
        x = torch.load(flags.checkpoint)
        begin_epoch = x['epoch']
        generator.load_state_dict(x['generator'])
        executor.load_state_dict(x['executor'])
        g_optimizer.load_state_dict(x['g_optimizer'])
        e_optimizer.load_state_dict(x['e_optimizer'])
    else:
        begin_epoch = 0

    if os.path.exists(flags.log):
        assert flags.checkpoint and os.path.exists(flags.checkpoint)
        text = open(flags.log).read()
        title = 'Epoch %d' % begin_epoch
        assert title in text

    predicted_whitespace = torch.zeros(
        flags.batch_size, len(g_alphabet), flags.g_width, device=device)
    predicted_whitespace[:, 0, :] = 1

    for epoch in range(begin_epoch, flags.num_epochs):
        gen_chr_losses = []
        gen_nul_losses = []
        gen_std_losses = []
        ok_losses = []
        exe_losses = []
        for batch in tqdm(list(range(flags.batches_per_epoch)), leave=False):
            g_optimizer.zero_grad()

            z = torch.randn(flags.batch_size, flags.z_dim, device=device)
            max_width = min(flags.g_width, 1 + int((epoch / 2) ** 0.5))
            widths = pick_widths(flags.batch_size, max_width)
            should_be_whitespace = widths_to_mask(widths, flags.g_width, device)
            code = generator(z, should_be_whitespace)
            softmax_code = F.softmax(code, 1)

            got_whitespace = softmax_code[:, 0, :]
            a = got_whitespace * should_be_whitespace
            b = (1 - got_whitespace) * (1 - should_be_whitespace)
            gen_chr_losses.append(a.mean().item())
            gen_nul_losses.append(b.mean().item())

            loss = -(a + b).abs().mean()
            loss.backward(retain_graph=True)

            c = softmax_code.mean(2).std(1) * 0.25
            gen_std_losses.append(c.mean().item())
            loss = c.mean()
            loss.backward(retain_graph=True)

            pred_output = executor(code)
            x = pred_output.max(1)[1]
            pred_out_text = e_dict.tensor_to_texts(x)
            true_output, in_text, true_out_text, oks = python(code)
            oks = torch.tensor(list(map(float, oks)), device=device)

            exe_loss = 1 - F.cross_entropy(pred_output, true_output)
            ok_loss = oks.mean()
            loss = exe_loss
            loss.backward()
            ok_losses.append(ok_loss.item())
            exe_losses.append(exe_loss.item())

            g_optimizer.step()

            e_optimizer.zero_grad()

            pred_output = executor(code.detach())
            x = pred_output.max(1)[1]
            pred_out_text = e_dict.tensor_to_texts(x)
            true_output, in_text, true_out_text, oks = python(code)

            loss = F.cross_entropy(pred_output, true_output)
            loss.backward()

            e_optimizer.step()

        gen_chr_loss = a.mean(1).tolist()
        gen_nul_loss = b.mean(1).tolist()
        gen_std_loss = c.tolist()
        exe_loss = F.cross_entropy(pred_output, true_output, reduction='none')
        ok_loss = ok_losses
        batch = {
            'gen_chr_loss': gen_chr_loss,
            'gen_nul_loss': gen_nul_loss,
            'gen_std_loss': gen_std_loss,
            'exe_loss': exe_loss,
            'ok_loss': ok_loss,
            'widths': widths,
            'in_text': in_text,
            'pred_out_text': pred_out_text,
            'true_out_text': true_out_text,
        }

        epoch = {
            'epoch': epoch,
            'gen_chr_loss': gen_chr_losses,
            'gen_nul_loss': gen_nul_losses,
            'gen_std_loss': gen_std_losses,
            'exe_loss': exe_losses,
            'ok_loss': ok_losses,
        }

        text = render_epoch_results(epoch, batch)
        with open(flags.log, 'a') as out:
            out.write(text + '\n')

        if flags.checkpoint:
            d = os.path.dirname(flags.checkpoint)
            if not os.path.exists(d):
                os.makedirs(d)
            x = {
                'epoch': epoch['epoch'],
                'generator': generator.state_dict(),
                'executor': executor.state_dict(),
                'g_optimizer': g_optimizer.state_dict(),
                'e_optimizer': e_optimizer.state_dict(),
            
            }
            torch.save(x, flags.checkpoint)


if __name__ == '__main__':
    main(parse_flags())
