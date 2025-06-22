from datasets import load_dataset
import pandas as pd
import json

# 加载所有 default 分区的 .parquet 文件
data_files = [f"/home/dell/yzt/openr1-math-220k/all/default-{i:05d}-of-00010.parquet" for i in range(10)]
dataset = load_dataset("parquet", data_files=data_files)["train"]

# 随机采样 1 万条数据
dataset = dataset.shuffle(seed=42).select(range(10000))
print(f"采样后数据集大小: {len(dataset)}")

# 转换为 JSON Lines 格式
output_file = "/home/dell/yzt/train.json"
with open(output_file, "w", encoding="utf-8") as f:
    for example in dataset:
        json.dump(example, f, ensure_ascii=False)
        f.write("\n")  # 每行一个 JSON 对象

print(f"已保存 1 万条数据到 {output_file}")

# 验证生成的 JSON 文件
df = pd.read_json(output_file, lines=True)
print(f"JSON 文件包含 {len(df)} 条记录")
print("示例记录:", df.iloc[0].to_dict())