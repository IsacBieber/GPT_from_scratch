import get_data
import remap
import torch
import torch.nn as nn
import save_model

# ====================model=============================================
class Bigram(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.w = nn.Linear(vocab_size, vocab_size, False)
    def forward(self, x):
        out = []
        for i in range(len(x)):
            out.append(self.w.weight[x[i]])
        out = torch.stack(out)
        return out

model = Bigram(remap.vocab_size)
# model.to("cuda")

# ====================loss function=====================================
def loss_func(pred, ans):
    # batch seq
    # 1.每个批次算一次loss -> 整个批次loss梯度降低的方向 -> maybe it's easy to learn
    # 2.整个批次算一次loss -> 整个批次loss梯度降低的方向 -> maybe it's hard to learn
    # pred.shape = [seq, vocab_size] ans.shape = [seq]
    loss = []
    for i in range(len(ans)):
        row = torch.exp(pred[i])
        sum = torch.sum(row)
        prob = row[ans[i]] / sum
        loss.append(-torch.log(prob))

    loss = torch.stack(loss)
    return loss.mean()


# ====================training=========================================
batch_size = 8
seq_size = 16
optim = torch.optim.Adam(model.parameters(), lr=0.001)

# ====================load model=======================================
start_epoch, last_loss = save_model.load_checkpoint(model, optim, "bigram.pt")

for i in range(0, 2000):
    x, y = get_data.get_batch(batch_size, seq_size)
    # x = x.to("cuda")
    # y = y.to("cuda")
    for j in range(batch_size):
        optim.zero_grad()
        pred = model.forward(x[j])
        loss = loss_func(pred, y[j])
        loss.backward()
        optim.step()
        if i % 500 == 0:
            print(loss.item())

    if i % 500 == 0:
        save_model.save_checkpoint(model, optim, i, loss, "bigram.pt")

# ======================reasoning=====================================
last_token = input("type your first token!\n")[0]
output_token = 200
print(last_token, end="")
for i in range(output_token):
    model.eval()
    with torch.no_grad():
        row = model.forward(torch.tensor([remap.char_to_int[last_token]]))[0]
    exp_row = torch.exp(row)
    sum = torch.sum(exp_row)
    prob = exp_row / sum
    idx = 0
    for j in range(len(prob)):
        if prob[j].item() == prob.max().item():
            idx = j

    print(remap.int_to_char[idx], end="")
    last_token = remap.int_to_char[idx]