import torch
import torch.nn as nn
from torch.nn import functional as F
import remap
import get_data
import save_model  # 引入我们强大的存档系统

# =================== 1. 硬件加速设定 ===================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.set_float32_matmul_precision('high')
print(f"🔥 当前使用的计算设备是: {device}")

# =================== 2. 疯狂拉大参数 ===================
vocab_size = remap.vocab_size 
batch_size = 64      
seq_size = 256       
embed_size = 384     
head_num = 6         
head_size = embed_size // head_num 
network_depth = 6    

# =================== 🚀 显存直读 DataLoader ===================
raw_data = torch.tensor(get_data.encoded, dtype=torch.long, device=device)

def fast_get_batch():
    ix = torch.randint(len(raw_data) - seq_size - 1, (batch_size,), device=device)
    offsets = torch.arange(seq_size, device=device)
    x = raw_data[ix.unsqueeze(1) + offsets]
    y = raw_data[ix.unsqueeze(1) + offsets + 1]
    return x, y

# =================== 3. 模型架构 =======================
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.Q = nn.Linear(embed_size, head_size, bias=False)
        self.K = nn.Linear(embed_size, head_size, bias=False)
        self.V = nn.Linear(embed_size, head_size, bias=False)

    def forward(self, x):
        q = self.Q(x) 
        k = self.K(x) 
        v = self.V(x) 
        output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(embed_size, embed_size) 

    def forward(self, x):
        almost_e = torch.cat([h(x) for h in self.heads], dim=-1) 
        final_e = self.proj(almost_e)
        return final_e

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embed_size)
        self.position_embedding_table = nn.Embedding(seq_size, embed_size)
        
        self.mha = MultiHeadAttention(head_num, head_size)
        
        self.FFNs = nn.ModuleList([nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.LayerNorm(embed_size)
        ) for _ in range(network_depth)])
        
        self.lm_head = nn.Linear(embed_size, vocab_size, bias=False)
        self.token_embedding_table.weight = self.lm_head.weight 

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) 
        
        x = self.token_embedding_table(idx) + self.position_embedding_table(pos)
        x = x + self.mha(x)  
        
        for FFN in self.FFNs:
            x = x + FFN(x) 
            
        logits = self.lm_head(x) 
        return logits

gpt = GPT().to(device)
opt = torch.optim.Adam(gpt.parameters(), lr=3e-4, fused=True) 

# =================== 4. 存档挂载与训练循环 =================
# 设定保存的文件名
model_filename = 'gpt_shakespeare.pt'

# 在开训前，尝试读取存档！
start_epoch, best_loss = save_model.load_checkpoint(gpt, opt, filename=model_filename, device=device)

# 如果你只想推理不想训练，可以把 total_epoch 改成和 start_epoch 一样大
total_epoch = 10000 

print(f"\n🚀 启动狂暴训练 (目标 Epoch: {total_epoch})...")

for i in range(start_epoch, total_epoch):
    x, y = fast_get_batch()
    
    opt.zero_grad(set_to_none=True)
    
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits = gpt(x) 
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        y = y.view(B*T)
        loss = F.cross_entropy(logits, y) 
    
    loss.backward()
    opt.step()

    # 每 500 轮打印一次并保存模型
    if i % 500 == 0:
        current_loss = loss.item()
        print(f"Epoch {i:5d} | Loss: {current_loss:.4f}")
        
        # 判断是不是历史最低 Loss
        is_best = current_loss < best_loss
        if is_best:
            best_loss = current_loss
            
        # 调用存档系统
        save_model.save_checkpoint(gpt, opt, i, current_loss, is_best, filename=model_filename)

# 跑完了最后再强制保存一次
save_model.save_checkpoint(gpt, opt, total_epoch, loss.item(), is_best=False, filename=model_filename)

# =================== 5. 推理部分 ===========================
def print_words(x_list):
    print("".join([remap.int_to_char[c] for c in x_list]))

prompt = input("\n📝 Type your prompt!\n")
output_token = 500 

context = torch.tensor([remap.char_to_int[c] for c in prompt], dtype=torch.long, device=device).unsqueeze(0)

print("============ 莎士比亚 2.0 生成开始 ============")
print_words(context[0].tolist())

gpt.eval()
with torch.no_grad():
    for _ in range(output_token):
        context_cond = context[:, -seq_size:]
        
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = gpt(context_cond)
            
        logits = logits[:, -1, :] 
        probs = F.softmax(logits, dim=-1)
        next_word = torch.multinomial(probs, num_samples=1)
        context = torch.cat((context, next_word), dim=1)
        
print_words(context[0].tolist())
print("============ 生成结束 ============")