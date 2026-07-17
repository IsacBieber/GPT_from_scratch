import torch
import torch.nn as nn
from torch.nn import functional as F
import remap
import get_data
import save_model

# =================== 1. 硬件加速设定 ===================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.set_float32_matmul_precision('high')
print(f"🔥 当前使用的计算设备是: {device}")

# =================== 2. 模型超参数 ===================
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

# =================== 3. 优化后的标准模型架构 =======================
class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_attn = nn.Linear(embed_size, 3 * embed_size, bias=False)
        self.c_proj = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(embed_size, dim=2)
        q = q.view(B, T, head_num, head_size).transpose(1, 2)
        k = k.view(B, T, head_num, head_size).transpose(1, 2)
        v = v.view(B, T, head_num, head_size).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_size)
        self.attn = CausalSelfAttention()
        self.ln_2 = nn.LayerNorm(embed_size)
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.GELU(), 
            nn.Linear(4 * embed_size, embed_size)
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embed_size)
        self.position_embedding_table = nn.Embedding(seq_size, embed_size)
        self.blocks = nn.ModuleList([Block() for _ in range(network_depth)])
        self.ln_f = nn.LayerNorm(embed_size) 
        self.lm_head = nn.Linear(embed_size, vocab_size, bias=False)
        self.token_embedding_table.weight = self.lm_head.weight 

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) 
        x = self.token_embedding_table(idx) + self.position_embedding_table(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) 
        return logits

gpt = GPT().to(device)
opt = torch.optim.Adam(gpt.parameters(), lr=3e-4, fused=True) 

# =================== 4. 存档挂载与训练循环 =================
model_filename = 'gpt_shakespeare.pt'

start_epoch, best_loss = save_model.load_checkpoint(gpt, opt, filename=model_filename, device=device)

if device == 'cuda':
    print("⚡ 正在编译模型，预计首次前向传播会有少许延迟...")
    gpt = torch.compile(gpt, backend="aot_eager")

total_epoch = 10000 
print(f"\n🚀 启动狂暴训练 (目标 Epoch: {total_epoch})...")

if start_epoch < total_epoch:
    for i in range(start_epoch, total_epoch):
        x, y = fast_get_batch()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = gpt(x) 
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)) 
        loss.backward()
        opt.step()

        if i % 500 == 0:
            current_loss = loss.item()
            print(f"Epoch {i:5d} | Loss: {current_loss:.4f}")
            is_best = current_loss < best_loss
            if is_best:
                best_loss = current_loss
            save_model.save_checkpoint(gpt, opt, i, current_loss, is_best, filename=model_filename)
    save_model.save_checkpoint(gpt, opt, total_epoch, loss.item(), is_best=False, filename=model_filename)
else:
    print(f"✅ 目标 Epoch ({total_epoch}) 已圆满达成！直接进入推理阶段...")

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
        
        # 🌟 加回了温度控制，防止过拟合变成缝合怪！
        temperature = 0.8  
        logits = logits / temperature 
        
        probs = F.softmax(logits, dim=-1)
        next_word = torch.multinomial(probs, num_samples=1)
        context = torch.cat((context, next_word), dim=1)
        
print_words(context[0].tolist())
print("============ 生成结束 ============")