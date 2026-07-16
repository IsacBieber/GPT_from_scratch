import remap
with open("data/encoded.txt", 'r', encoding="utf-8") as f:
    encoded_str = f.read()

remap = [int(x) for x in encoded_str.split(',')]

for i in range(min(50, len(remap))):
    print(remap.int_to_char[remap[i]])