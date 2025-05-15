#!/usr/bin/env python
import os
import sys
import tensorflow as tf
import functools

# 导入magenta训练模块和模型模块
try:
    from magenta.models.music_vae import music_vae_train as train_script
    # 新增：导入模型模块，用于冻结层
    import magenta.models.music_vae.trained_model as trained_model
    print("成功导入Magenta模块")
except ImportError as e:
    print(f"无法导入Magenta模块: {e}")
    print("请确保已正确安装magenta库")
    sys.exit(1)

# 检查TensorFlow版本
tf_version = tf.__version__
print(f"当前TensorFlow版本: {tf_version}")
if not tf_version.startswith("2."):
    print("警告: 此版本针对TensorFlow 2.x，您的版本可能不兼容")

def freeze_layers_for_digiscore(model):
    """为数字乐谱项目冻结特定层 - 适配TensorFlow 2.x
    
    冻结策略：
    1. 编码器: 只冻结第一层BiLSTM/RNN
    2. 解码器: 冻结层次解码器的基础层
    3. 保持潜在空间完全可训练
    """
    print("\n数字乐谱(DigiScore)项目 - 冻结层方案:")
    print("---------------------------------------------")
    
    frozen_count = 0
    trainable_count = 0
    
    # 按类别统计参数
    encoder_layers = 0
    decoder_layers = 0
    latent_layers = 0
    other_layers = 0
    
    # 首先打印所有层以便调试
    print("模型所有层名:")
    for i, layer in enumerate(model.layers):
        print(f"{i}: {layer.name}")
    
    # 遍历所有层，应用冻结策略
    for layer in model.layers:
        if "encoder" in layer.name:
            encoder_layers += 1
            # 只冻结编码器的第一层 (可能是bilstm_0, rnn_0等)
            if any(x in layer.name for x in ["bilstm_0", "rnn_0", "rnn_cell_0"]):
                layer.trainable = False
                frozen_count += 1
                print(f"已冻结编码器层: {layer.name}")
            else:
                trainable_count += 1
        elif "decoder" in layer.name:
            decoder_layers += 1
            # 冻结解码器的基础层
            if any(x in layer.name for x in ["core_decoder_0", "output_projection_0", "rnn_cell_0/level_0"]):
                layer.trainable = False
                frozen_count += 1
                print(f"已冻结解码器层: {layer.name}")
            else:
                trainable_count += 1
        elif any(x in layer.name for x in ["z_", "latent"]):
            # 保持潜在空间完全可训练
            latent_layers += 1
            trainable_count += 1
        else:
            other_layers += 1
            trainable_count += 1
    
    print("\n冻结层统计:")
    print(f"- 总层数: {frozen_count + trainable_count}")
    print(f"- 已冻结: {frozen_count}")
    print(f"- 可训练: {trainable_count}")
    print(f"- 编码器层: {encoder_layers}")
    print(f"- 解码器层: {decoder_layers}")
    print(f"- 潜在空间层: {latent_layers}")
    print(f"- 其他层: {other_layers}")
    print("---------------------------------------------")
    print("冻结策略：保留低层特征提取能力，同时优化高层特征和MIDI CC控制参数生成")
    print("这种配置最适合在good/bad样本之间进行插值生成数字乐谱\n")

def main():
    """主函数 - 应用冻结层策略并启动训练"""
    
    # 保存原始的TrainedModel.__init__
    original_init = trained_model.TrainedModel.__init__
    
    # 定义新的__init__，添加冻结层逻辑
    def patched_init(self, *args, **kwargs):
        # 先调用原始初始化
        original_init(self, *args, **kwargs)
        # 确保模型已创建
        if hasattr(self, '_model'):
            print("模型已创建，准备应用层冻结...")
            # 应用我们的冻结逻辑
            freeze_layers_for_digiscore(self._model)
        else:
            print("警告：无法获取模型实例，冻结层失败")
    
    # 替换原始__init__，这样每次创建模型时都会自动应用冻结
    trained_model.TrainedModel.__init__ = patched_init
    print("已设置DigiScore冻结层钩子...")
    
    # 启动训练
    print("开始训练...")
    train_script.console_entry_point()

if __name__ == "__main__":
    main() 