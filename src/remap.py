with open("data/shakespeare.txt", 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

char_to_int = {}
for i, c in enumerate(chars):
    char_to_int[c] = i

int_to_char = {}
for i, c in enumerate(chars):
    int_to_char[i] = c

encoded = []
for c in text:
    encoded.append(char_to_int[c])

# f里面只能写入字符串，所以str(x)
with open("data/encoded.txt", 'w', encoding='utf-8') as f:
    f.write(','.join([str(x) for x in encoded]))
