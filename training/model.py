import torch
import torch.nn as nn
from transformers import CLIPVisionModel

class ForensicsAdapter(nn.Module):
    """
    参考 Forensics Adapter 论文与杨老师会议指导：
    构建双通道架构，主通道提取全局特征，残差通道捕捉局部边缘篡改细节。
    """
    def __init__(self):
        super().__init__()
        
        # 1. 主通道 (Global Channel)：加载预训练的 CLIP 视觉模型
        # 小白防坑：直接用现成的预训练权重，不要自己从头训练！
        self.clip = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        
        # 按照论文要求，冻结 CLIP 的所有参数，保证它原有的强大通用性不被破坏 
        for param in self.clip.parameters():
            param.requires_grad = False
            
        # 2. 残差通道 (Residual Channel)：轻量级 Adapter
        # 杨老师强调：用残差信号提升隐写痕迹的检测效果。这里用一个小型卷积网络模拟 ViT-tiny。
        self.adapter = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)) # 浓缩特征
        )
        
        # 3. 信号融合层 (Cross-Attention Fusion)
        # 核心创新点：杨老师明确要求“以残差为查询(Query)、整图CLIP特征为键值(Key)进行引导增强”
        self.fusion_attention = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
        self.adapter_proj = nn.Linear(128 * 7 * 7, 768) # 维度对齐层，把 Adapter 输出对齐到 768维
        
        # 4. 输出预测头：2分类 (输出 0代表真，1代表假)
        self.classifier = nn.Linear(768, 2)

    def forward(self, x):
        # x 是输入的图片，形状大概是 [batch_size, 3, 224, 224]
        
        # --- 步骤 A: 主通道获取 CLIP 特征 ---
        clip_outputs = self.clip(pixel_values=x)
        clip_features = clip_outputs.pooler_output # 提取全局特征 [batch_size, 768]
        clip_features = clip_features.unsqueeze(1) # 增加一个维度，变成 [batch_size, 1, 768]
        
        # --- 步骤 B: 残差通道获取 局部/高通滤波 特征 ---
        adapter_features = self.adapter(x)
        adapter_features = adapter_features.view(adapter_features.size(0), -1) # 展平操作
        adapter_features = self.adapter_proj(adapter_features).unsqueeze(1) # 维度对齐为 [batch_size, 1, 768]
        
        # --- 步骤 C: 交叉注意力融合 ---
        # 严格执行杨老师方案：Adapter 特征做 Query，CLIP 特征做 Key 和 Value
        fused_features, _ = self.fusion_attention(
            query=adapter_features, 
            key=clip_features, 
            value=clip_features
        )
        
        fused_features = fused_features.squeeze(1) # 降维
        
        # --- 步骤 D: 判定真假 ---
        logits = self.classifier(fused_features)
        return logits

if __name__ == "__main__":
    print("正在构建符合杨老师要求的 Forensics Adapter 双通道模型...")
    model = ForensicsAdapter()
    print("模型构建成功！已挂载冻结的 CLIP 主通道 和 可训练的残差 Adapter 融合通道。")
    
    # 我们用一段随机生成的“假数据”测试一下模型能不能跑通
    print("输入模拟图像进行测试跑通验证...")
    dummy_input = torch.randn(1, 3, 224, 224) # 模拟1张 3通道的 224x224 图像
    output = model(dummy_input)
    print(f"测试成功！输出形状为: {output.shape} (预期为 [1, 2]，代表真、假两个类别的打分)")