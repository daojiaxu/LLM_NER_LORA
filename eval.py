import json
import random
import torch
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 加载模型和分词器
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("./qwen/Qwen2-1___5B-Instruct/", use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("./qwen/Qwen2-1___5B-Instruct/", device_map="auto", torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, model_id="./output/Qwen2-NER/checkpoint-884")
    return model, tokenizer

def predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def load_data(file_path, sample_size=0.1):
    """
    从 jsonl 文件加载数据，并选取一定比例的样本
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    # 随机选取 10% 的数据
    sample_data = random.sample(data, int(len(data) * sample_size))
    return sample_data

import json

def extract_entities(text):
    """提取实体及其标签，处理可能多个连续的JSON对象"""
    entities = []
    try:
        # 如果是多个连续的 JSON 对象，把它们包装成一个数组
        if text.startswith("{") and not text.startswith("["):
            # 将多个 JSON 对象合并为一个 JSON 数组
            text = "[" + text.replace("}{", "},{") + "]"
        # 解析 JSON 格式
        entity_list = json.loads(text)
        if isinstance(entity_list, list):
            for entity in entity_list:
                if isinstance(entity, dict) and "entity_text" in entity and "entity_label" in entity:
                    # 添加实体文本和标签
                    entities.append({
                        "entity_text": entity["entity_text"],
                        "entity_label": entity["entity_label"]
                    })
    except json.JSONDecodeError:
        print(f"JSON 解码错误: {text}")  # 调试输出
        pass
    return entities

def evaluate(predictions, true_labels):
    """
    计算 Precision, Recall, F1
    计算基于实体文本和标签的匹配
    """
    # 将预测和真实标签转换为 (实体文本, 实体标签) 的元组列表
    predicted_entities = [(e[0], e[1]) for e in predictions]  # 预测的实体（实体文本和标签）
    true_entities = [(e[0], e[1]) for e in true_labels]  # 真实的实体（实体文本和标签）

    print("预测实体：",predicted_entities)
    print("真实实体：",true_entities)

    # 计算 True Positives (TP), False Positives (FP), False Negatives (FN)
    tp = len([e for e in predicted_entities if e in true_entities])  # 预测正确的实体
    fp = len([e for e in predicted_entities if e not in true_entities])  # 错误预测的实体
    fn = len([e for e in true_entities if e not in predicted_entities])  # 漏检的实体
    print("TP:", tp)
    print("FP:", fp)
    print("FN:", fn)


    # 计算 Precision, Recall 和 F1 分数
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1

def main():
    # 加载模型和分词器
    model, tokenizer = load_model()

    # 加载数据
    file_path = "data/ccf_train.jsonl"
    test_data = load_data(file_path, sample_size=0.002)

    true_labels = []
    predictions = []
    i = 0
    for row in test_data:
        i += 1
        print(i, "/", len(test_data))
        instruction = row["instruction"]
        input_value = row["input"]
        true_entities = row["output"]  # 假设真实的实体标签在 `output` 字段中

        # 构建消息格式
        messages = [
            {"role": "system", "content": f"{instruction}"},
            {"role": "user", "content": f"{input_value}"}
        ]

        # 获取模型预测的响应
        response = predict(messages, model, tokenizer)

        # 提取实体：真实标签和预测标签
        predicted_entities = extract_entities(response)  # 提取预测实体
        true_entity_list = extract_entities(true_entities)  # 提取真实实体标签

        # 存储真实标签和预测标签（实体文本和标签一起存储）
        true_labels.extend([(e["entity_text"], e["entity_label"]) for e in true_entity_list])  # 真实标签
        predictions.extend([(e["entity_text"], e["entity_label"]) for e in predicted_entities])  # 预测标签

        # 可选，打印示例输出（帮助调试）
        print(f"预测: {response}")
        print(f"真实: {true_entities}\n")

    # 调试：打印实际的 true_labels 和 predictions 数量
    print(f"Total true labels: {len(true_labels)}")
    print(f"Total predictions: {len(predictions)}")

    # 计算评估指标
    precision, recall, f1 = evaluate(predictions, true_labels)

    # 输出评估结果
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")


if __name__ == "__main__":
    main()
