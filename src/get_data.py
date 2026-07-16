import remap
import torch
import random

with open("data/encoded.txt", 'r', encoding="utf-8") as f:
    encoded_str = f.read()
encoded = [int(x) for x in encoded_str.split(',')]

cur = []
nxt = []

for i in range(len(encoded)):
    cur.append(encoded[i])
    if i + 1 < len(encoded) :
        nxt.append(encoded[i + 1])
    else :
        nxt.append(remap.char_to_int['!'])

def get_batch(batch_size = 4, seq_size = 8) :
    x, y = [], []
    for i in range(batch_size) :
        begin = random.randint(0, len(cur) - seq_size - 1)
        x.append([cur[begin + j] for j in range(seq_size)])
        y.append([nxt[begin + j] for j in range(seq_size)])

    return torch.tensor(x), torch.tensor(y)
# print(len(encoded)) 100w

# for c in cur:
#     print(c, end=" ")

# print("\n");

# for c in nxt:
#     print(c, end=" ")

