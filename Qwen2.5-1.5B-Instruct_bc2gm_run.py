import json
import pandas as pd
import os
from transformers import PreTrainedTokenizerFast
from modelscope import snapshot_download, AutoTokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from swanlab.integration.huggingface import SwanLabCallback
import swanlab
import logging
from tqdm import tqdm
import re

### 数据集：bc2gm_data

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# 加载、处理数据集和测试集
# train_jsonl_new_path = "biomed_ner_data/formatted_train.json"

train_df = pd.read_json("bc2gm_data/train_formatted.json")
test_df = pd.read_json("bc2gm_data/test_formatted.json")

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
# model_dir = "./qwen/Qwen2-1___5B-Instruct"

# 在modelscope上下载Qwen模型到本地目录下
model_dir = snapshot_download(model_id, cache_dir="./", revision="master")
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir, use_fast=True, trust_remote_code=True)


# Transformers加载模型权重
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.bfloat16)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法


# 数据处理函数
def process_func(example):
    """
    将数据集进行预处理, 处理成模型可以接受的格式
    """
    MAX_LENGTH = 600
    input_ids, attention_mask, labels = [], [], []
    system_prompt = """You are an expert in biomedical entity recognition. Given the text, your task is to extract specific entities from the following categories: Chemicals, Clinical Drugs, Body Substances, Anatomical Structures, Cells and Their Components, Genes and Gene Products, Intellectual Property, Language, Regulation or Law, Geographical Areas, Organisms, Groups, People, Organizations, Products, Locations, Phenotypes, Disorders, Signaling Molecules, Events, Medical Procedures, Activities, Functions, and Money. Output the recognized entities in JSON format, with each entity represented as {\"entity_text\": \"text\", \"entity_label\": \"category\"}. Note: 1. Each entity must be output as a correct JSON string. 2. If no entities are found, output \"No entities found\"."""

    instruction = tokenizer(
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = (
            instruction["attention_mask"] + response["attention_mask"] + [1]
    )
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# 加载训练数据
train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)

# PEFT配置
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alpha
    lora_dropout=0.1,  # Dropout 比例
)

model = get_peft_model(model, config)

# 训练配置
args = TrainingArguments(
    output_dir="output/Qwen2.5-1.5B-Instruct_bc2gm_model",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=6,
    save_steps=200,  # 每100步保存一次checkpoint
    eval_steps=200,  # 每100步评估一次
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)

# SwanLab回调
swanlab_callback = SwanLabCallback(
    project="Qwen2-NER-fintune",
    experiment_name="Qwen2.5-1.5B-Instruct-NER_bc2gm_data",
    description="使用Qwen2.5-1.5B-Instruct模型在bc2gm_data数据集上微调，实现关键实体识别任务。",
    config={
        "model": model_id,
        "model_dir": model_dir,
        "dataset": "_bc2gm_data",
    },
)

def extract_entities(text):
    """
    提取实体及其标签，处理可能多个连续的JSON对象，包括复杂的残缺数据。
    参数:
    text (str): 包含JSON格式实体信息的字符串。
    返回:
    list: 包含字典的列表，每个字典包含'entity_text'和'entity_label'键。
    """
    # 使用正则表达式匹配完整的 JSON 对象
    pattern = r'\{.*?\}'
    matches = re.findall(pattern, text)

    entities = []

    for match in matches:
        try:
            # 解析 JSON 格式
            entity = json.loads(match)
            if isinstance(entity, dict) and "entity_text" in entity and "entity_label" in entity:
                # 添加实体文本和标签
                entities.append({
                    "entity_text": entity["entity_text"],
                    "entity_label": entity["entity_label"]
                })
        except json.JSONDecodeError as e:
            print(f"JSON 解码错误: {e} - 输入数据: {match}")  # 调试输出
            print(f"JSON 解码错误: {e} - text数据: {text}")  # 调试输出

    return entities


def evaluate(predictions, true_labels):
    """
    计算 Precision, Recall, F1
    计算基于实体文本和标签的匹配
    """
    # 将预测和真实标签转换为 (实体文本, 实体标签) 的元组列表
    predicted_entities = [(e["entity_text"], e["entity_label"]) for e in predictions if isinstance(e, dict)]
    true_entities = [(e["entity_text"], e["entity_label"]) for e in true_labels if isinstance(e, dict)]

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


def predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# 评估整个测试集
def evaluate_model(test_df):
    predictions = []
    true_labels = []

    # Wrap the loop with tqdm to show progress
    for index, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating", unit="row"):
        instruction = row["instruction"]
        input_value = row["input"]
        true_entities = row["output"]

        messages = [
            {"role": "system", "content": f"{instruction}"},
            {"role": "user", "content": f"{input_value}"}
        ]

        response = predict(messages, model, tokenizer)
        predicted_entities = extract_entities(response)
        true_entity_list = extract_entities(true_entities)

        true_labels.extend(true_entity_list)
        predictions.extend(predicted_entities)

        logger.debug(f"预测: {response}")
        logger.debug(f"真实: {true_entities}\n")
        # print(f"预测: {response}")
        # print(f"真实: {true_entities}\n")
    precision, recall, f1 = evaluate(predictions, true_labels)

    logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return precision, recall, f1

# 自定义评估回调，确保每个checkpoint时评估
from transformers import TrainerCallback

class EvalCallback(TrainerCallback):
    def __init__(self, test_df, tokenizer, model):
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.model = model

    def on_init_end(self, args, state, control, **kwargs):
        # This method will be called after the Trainer's initialization
        logger.info("Trainer initialized. Starting the training process.")

    def on_epoch_end(self, args, state, control, **kwargs):
        # This method will be called at the end of each epoch
        logger.info(f"End of epoch {state.epoch}. Evaluation will be done.")

    def on_step_end(self, args, state, control, **kwargs):
        # This method will be called after each training step
        if state.global_step % args.eval_steps == 0:
            logger.info(f"\nEvaluation at step {state.global_step}")
            precision, recall, f1 = evaluate_model(self.test_df)
            logger.info(f"Step {state.global_step} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback, EvalCallback(test_df, tokenizer, model)],
)

trainer.train()


# ====== 训练结束后的预测 ===== #

# 使用新的 `evaluate_model` 方法进行评估
evaluate_model(test_df)
