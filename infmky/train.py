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
    a.add_argument('--batch_size', type=int, default=16)
    a.add_argument('--z_dim', type=int, default=256)
    a.add_argument('--g_width', type=int, default=16)
    a.add_argument('--e_width', type=int, default=32)
    a.add_argument('--checkpoint', type=str, default='data/checkpoint.pt')
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
            r = 'SyntaxError %d:%d' % (e.lineno, e.offset)
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
        in_text = '\n'.join(ss)
        rr = []
        oks = []
        for s in ss:
            r, ok = python(s)
            r = r.ljust(self.e_width)[:self.e_width]
            rr.append(r)
            oks.append(ok)
        out_text = '\n'.join(rr)
        r = self.e_dict.texts_to_tensor(rr)
        return r, in_text, out_text, oks


def main(flags):
    device = torch.device(flags.device)

    g_alphabet = string.digits + '+-*=> '
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

    g_params = list(generator.parameters())
    e_params = list(executor.parameters())
    optimizer = Adam(g_params + e_params)

    if flags.checkpoint and os.path.exists(flags.checkpoint):
        x = torch.load(flags.checkpoint)
        begin_epoch = x['epoch']
        generator.load_state_dict(x['generator'])
        executor.load_state_dict(x['executor'])
        optimizer.load_state_dict(x['optimizer'])
    else:
        begin_epoch = 0

    for epoch in range(begin_epoch, flags.num_epochs):
        losses = []
        for batch in tqdm(list(range(flags.batches_per_epoch)), leave=False):
            optimizer.zero_grad()

            z = torch.randn(flags.batch_size, flags.z_dim, device=device)
            code = generator(z)
            pred_output = executor(code)
            true_output, in_text, out_text, oks = python(code)

            loss = F.cross_entropy(pred_output, true_output)
            loss.backward()
            losses.append(loss.item())

            for x in generator.parameters():
                x.grad.data.mul_(-1)

            optimizer.step()

        print('=' * 80)
        print('epoch %d loss: mean %.3f std %.3f' %
              (epoch, np.mean(losses), np.std(losses)))
        print()
        print(in_text)
        print()
        print(out_text)
        print()

        if flags.checkpoint:
            d = os.path.dirname(flags.checkpoint)
            if not os.path.exists(d):
                os.makedirs(d)
            x = {
                'epoch': epoch,
                'generator': generator.state_dict(),
                'executor': executor.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(x, flags.checkpoint)


if __name__ == '__main__':
    main(parse_flags())
