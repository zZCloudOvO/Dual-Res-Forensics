import torch
import torch.nn as nn
import torch.optim as optim
# 从你刚刚写的 training 文件夹里导入我们搭建好的双通道模型
from training.model import ForensicsAdapter

def main():
    print("="*50)
    print("🚀 启动 DeepfakeBench 模块化训练框架")
    print("="*50)

    # 1. 实例化模型
    print("[1/4] 正在加载 Forensics Adapter 双通道模型...")
    # 注意：第一次运行会自动从 HuggingFace 下载 CLIP 预训练权重，大概需要一两分钟，请耐心等待
    model = ForensicsAdapter()
    
    # 将模型放入 GPU (如果有的话)，小白没有 GPU 也可以用 CPU 跑通逻辑
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"✅ 模型加载完毕，当前运行设备: {device}")

    # 2. 设置优化器和损失函数
    # 遵循 DeepfakeBench 规范：使用 Adam 优化器，学习率设为 0.0002 
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    criterion = nn.CrossEntropyLoss()
    print("[2/4] 优化器 (Adam) 和 损失函数 (CrossEntropy) 配置完毕。")

    # 3. 模拟数据加载 (A/B 图像集)
    # 杨老师提出通过构建A（全局一致）和B（局部不一致）图像集来凸显属性差异
    print("[3/4] 正在生成模拟训练数据 (代表 A集-真脸 和 B集-假脸)...")
    batch_size = 4
    # 模拟 4 张 3通道 224x224 的图片
    dummy_images = torch.randn(batch_size, 3, 224, 224).to(device)
    # 模拟这 4 张图片的标签：比如 0代表真(A集)，1代表假(B集)
    dummy_labels = torch.tensor([0, 1, 0, 1]).to(device) 

    # 4. 启动简易训练循环 (Training Loop)
    print("[4/4] 开始训练循环...")
    epochs = 3 # 为了快速验证，我们只跑 3 轮
    
    model.train() # 设置模型为训练模式
    for epoch in range(epochs):
        optimizer.zero_grad() # 清空过往梯度
        
        # 前向传播：把图片输入模型，得到预测打分
        outputs = model(dummy_images)
        
        # 计算损失 (预测值和真实标签的差距)
        loss = criterion(outputs, dummy_labels)
        
        # 反向传播，更新残差通道(Adapter)的参数
        loss.backward()
        optimizer.step()
        
        print(f" -> Epoch [{epoch+1}/{epochs}] | Loss: {loss.item():.4f} | 状态: 正常运行中")

    print("="*50)
    print("🎉 恭喜！基础训练框架测试跑通！")
    print("下一步计划：对接真实的 FF++ 数据集，引入图像过渡区域的 Token 引导机制。")
    print("="*50)

if __name__ == "__main__":
    main()