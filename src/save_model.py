# src/save_model.py
import os
import torch
import shutil

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)  

def save_checkpoint(model, optimizer, epoch, loss, is_best=False, filename='checkpoint.pt'):
    filepath = os.path.join(MODELS_DIR, filename)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    
    if is_best:
        best_filepath = os.path.join(MODELS_DIR, 'best_' + filename)
        shutil.copyfile(filepath, best_filepath)
        print(f"🌟 [存档系统] 发现历史最低 Loss: {loss:.4f}！已自动备份为最佳模型: {best_filepath}")

def load_checkpoint(model, optimizer, filename='checkpoint.pt', device='cpu'):
    filepath = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(filepath):
        print("🆕 [存档系统] 未找到历史存档，将从 0 开始训练全新模型...")
        return 0, float('inf') 

    checkpoint = torch.load(filepath, map_location=device)
    
    # 提取模型权重字典
    state_dict = checkpoint['model_state_dict']
    
    # 🛠️ 【核心修复：脱水器】去除 torch.compile 带来的 _orig_mod. 前缀
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            # 把带有前缀的键删掉，换成没有前缀的键
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    # 加载脱水后的干净权重
    model.load_state_dict(state_dict)
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    start_epoch = checkpoint['epoch'] + 1   
    last_loss = checkpoint['loss']
    print(f"🔄 [存档系统] 成功读取读档 {filepath}！(已自动清理编译前缀)")
    print(f"   >>> 进度恢复至 Epoch {start_epoch}，上次存档 Loss: {last_loss:.4f}")
    
    return start_epoch, last_loss