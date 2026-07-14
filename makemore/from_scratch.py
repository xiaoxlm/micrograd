# python makemore/from_scratch.py
import torch

words = open('makemore/name.txt', 'r').read().splitlines()

N = torch.zeros((28, 28), dtype=torch.int32)
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1


import matplotlib.pyplot as plt

plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha='center', va='bottom', color='gray')
        plt.text(j, i, N[i, j].item(), ha='center', va='top', color='gray')

plt.axis('off')
#plt.show()

P = N.float()
P /= P.sum(1, keepdims=True)

g = torch.Generator().manual_seed(2147483647)

for i in range(10):
    idx = 0
    out = []
    while True:
        # p = N[idx].float()
        # p = p / p.sum()
        p = P[idx]
        idx = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[idx])
        if idx == 0:
            break
    print(''.join(out))

# GOAL: maximize likelihood of the data w.r.t. model parameters(statistical modeling)
# equivalent to maximizing the log likelihood(because log is monotonic)
# equivalent to minimizing the negative log likelihood
# equivalent to minimizing the average negative log likelihood

log_likelihood = 0.0
n = 0
for w in words[:3]:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        log_likelihood += torch.log(prob)
        n += 1
        print(f'{ch1}{ch2}: {prob:.4f} {log_likelihood:.4f}')
print(log_likelihood / n)
nll = -log_likelihood
print(f'{nll=}')
print(f'{nll/n}')
