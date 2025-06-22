from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 设置模型名称
model_name = "Qwen/Qwen2.5-14B-Instruct"

# 下载并加载分词器
print("正在下载分词器...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)
print("分词器下载完成！")

# 下载并加载模型
print("正在下载模型...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # 使用 bfloat16 节省显存
    device_map="auto",          # 自动分配到可用设备（GPU/CPU）
    trust_remote_code=True
)
print("模型下载完成！")

# 保存模型和分词器到本地（可选）
save_directory = "./qwen2.5"
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)
print(f"模型和分词器已保存到 {save_directory}")

