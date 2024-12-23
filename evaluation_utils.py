import json
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging
import logging
from tqdm import tqdm

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
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
        # print(f"JSON 解码错误: {text}")  # 调试输出
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

    precision, recall, f1 = evaluate(predictions, true_labels)

    logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return precision, recall, f1

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