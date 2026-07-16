import torch.nn as nn
import torch
import remap
import get_data

# ===================procudure ========================
# next token prediction:
# last word in x ----(embedding matrix)---> embedding vector e (shape = (1, A))
# e do matrix multiplication with Q, K, V (shape = (A, B)) -----> q, k, v of e (shape = (1, B))
# e's q do dot product with every previous word's k --------> get a w (shape = (curent_lenth, B))
# w do matrix multiplication with previous concatenate v --------> new_e(shape = (1, B))
# for multihead attention, concatenate multi new_e to get a almost_e (shape = (1, A))
# almost_e do matrix multiplication with a linear layer -------> final_e (shape = (1, A))
# final_e + e (残差链接) and dive into FNN -> residual -> FNN loop

# ===================custom Layer =====================
class Linear_layer(nn.Module):
    def __init__(self, row, column):
        super().__init__()
        self.W = nn.Parameter(torch.randn(row, column) * 0.1)
        self.B = nn.Parameter(torch.zeros(column))
    def forward(self, x):
        return x @ self.W + self.B

def softmax(x):
    x_max = torch.max(x)
    exp_x = torch.exp(x - x_max)
    # 减去最大值比例不变 => 概率不变??
    return exp_x / torch.sum(exp_x)

class SelfAttention(nn.Module):
    # 【改动1】：__init__ 增加 pos_matrix 接收位置矩阵
    def __init__(self, row, column, embedding_matrix, pos_matrix):
        super().__init__()
        self.embedding_matrix = embedding_matrix
        self.pos_matrix = pos_matrix
        self.row = row
        self.column = column
        self.Q = Linear_layer(row, column)
        self.K = Linear_layer(row, column)
        self.V = Linear_layer(row, column)

    # 【改动2】：提取向量时，同时传入字(token)和它的位置索引(pos_index)，两者相加
    def embedding_vector(self, token, pos_index):
        return self.embedding_matrix.W[token] + self.pos_matrix[pos_index]

    def get_q(self, token, pos_index):
        return self.Q(self.embedding_vector(token, pos_index))

    def get_k(self, token, pos_index):
        return self.K(self.embedding_vector(token, pos_index))

    def get_v(self, token, pos_index):
        return self.V(self.embedding_vector(token, pos_index))

    def forward(self, x):
        # 【改动3】：调用时传入位置索引。x[-1] 的位置是 len(x)-1
        q = self.get_q(x[-1], len(x) - 1)
        w = torch.zeros(len(x))
        for i in range(len(x)):
            # 这里点积会爆炸，加上这个优化，能干啥，好像能稳一点
            w[i] = torch.dot(q, self.get_k(x[i], i)) / (self.column ** 0.5)

        w = softmax(w)

        output = torch.zeros(self.column)
        for i in range(len(w)):
            output += self.get_v(x[i], i) * w[i]

        return output


# ===================define model ======================
vocab_size = remap.vocab_size # 65
embed_size = 32
attention_size = 32
head_num = 1
header_size = embed_size // head_num
network_depth = 2

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_matrix = Linear_layer(vocab_size, embed_size)
        
        # 【改动4】：新建一个位置矩阵，假设你的句子最长不超过 1024
        self.pos_matrix = nn.Parameter(torch.randn(1024, embed_size) * 0.1)
        
        self.selfattention = SelfAttention(embed_size, attention_size, self.embedding_matrix, self.pos_matrix)
        
        self.FFNs = nn.ModuleList([nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.LayerNorm(embed_size)
        ) for i in range(network_depth)])
        
    def forward(self, x):
        # x: original tokens
        input = self.selfattention(x)
        output = input
        for FFN in self.FFNs:
            output = output + FFN(output)
        return output @ self.embedding_matrix.W.T

gpt = GPT()


# ===================define loss function ===============
def loss_function(predict, real):
    prob = torch.stack([softmax(predict[i]) for i in range(len(predict))])

    loss_sum = 0
    for i in range(len(predict)):
        loss_sum += -torch.log(prob[i][real[i]] + 1e-8)
    return loss_sum / len(real)


# ===================optimizer ==========================
opt = torch.optim.Adam(gpt.parameters(), lr=0.0003)


# ===================training ===========================
epoch = 10000
batch_size = 2
seq_size = 16
for i in range(epoch):
    opt.zero_grad()

    x, y = get_data.get_batch(batch_size, seq_size)
    loss = 0
    for j in range(batch_size):
        predict = []
        for k in range(len(x[j])):
            predict.append(gpt.forward(x[j][0:k + 1:1]))
        predict = torch.stack(predict)
        loss += loss_function(predict, y[j])
        
    loss = loss / batch_size
    loss.backward()
    opt.step()

    if i % 500 == 0:
        print(loss.item())

# ===================reasoning ===========================
def print_words(x):
    x = [remap.int_to_char[c] for c in x]
    for c in x:
        print(c, end="")
    print("\n")

context = input("type your prompt!\n")
output_token = 100
context = [remap.char_to_int[c] for c in context]
print_words(context)

for i in range(output_token):
    gpt.eval()
    with torch.no_grad():
        predict = softmax(gpt.forward(context))
        
        # 【改动5】：用多项式采样（掷骰子）彻底替换掉你原本那段找 max() 的 for 循环
        next_word = torch.multinomial(predict, num_samples=1).item()
        
        context.append(next_word)
        print_words(context)