# src/save_model.py
import os
import torch
import shutil

# 自动定位到 src 的上级目录下的 models 文件夹
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)  # 没有就自动创建

def save_checkpoint(model, optimizer, epoch, loss, is_best=False, filename='checkpoint.pt'):
    """
    保存模型参数、优化器状态等。
    如果 is_best 为 True，还会额外复制一份作为 best_model.pt
    """
    filepath = os.path.join(MODELS_DIR, filename)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    # 先保存当前的最新进度
    torch.save(checkpoint, filepath)
    
    # 如果是历史最佳，额外保存一份专属文件
    if is_best:
        best_filepath = os.path.join(MODELS_DIR, 'best_' + filename)
        shutil.copyfile(filepath, best_filepath)
        print(f"🌟 [存档系统] 发现历史最低 Loss: {loss:.4f}！已自动备份为最佳模型: {best_filepath}")

def load_checkpoint(model, optimizer, filename='checkpoint.pt', device='cpu'):
    """
    加载模型。如果找不到文件，返回 0 和 无穷大Loss（代表从头训练）
    """
    filepath = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(filepath):
        print("🆕 [存档系统] 未找到历史存档，将从 0 开始训练全新模型...")
        return 0, float('inf') # 初始最佳 loss 设为无穷大

    # 加载存档
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 如果优化器状态不为空才加载 (防止更换优化器后报错)
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    start_epoch = checkpoint['epoch'] + 1   # 从存档的下一轮开始
    last_loss = checkpoint['loss']
    print(f"🔄 [存档系统] 成功读取读档 {filepath}！")
    print(f"   >>> 进度恢复至 Epoch {start_epoch}，上次存档 Loss: {last_loss:.4f}")
    
    return start_epoch, last_loss