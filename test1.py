import json

# 读取 JSON 文件
file_path = 'biomed_ner_data/processed_train.json'

with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)


# 创建输出格式的函数
def format_entities(sentence, entities):
    # 构建每个实体的格式：{"entity_text": "text", "entity_label": "category"}
    entity_output = []
    for entity in entities:
        entity_text = sentence[entity['start']:entity['end']]
        entity_label = entity['class']
        entity_output.append(f'{{"entity_text": "{entity_text}", "entity_label": "{entity_label}"}}')

    return ' '.join(entity_output) if entity_output else '"No entities found"'


# 创建输出的 JSON 格式
formatted_data = []
for entry in data:
    sentence = entry['sentence']
    entities = entry['entities']

    formatted_entry = {
        "instruction": "You are an expert in biomedical entity recognition. Given the text, your task is to extract specific entities from the following categories: Chemicals, Clinical Drugs, Body Substances, Anatomical Structures, Cells and Their Components, Genes and Gene Products, Intellectual Property, Language, Regulation or Law, Geographical Areas, Organisms, Groups, People, Organizations, Products, Locations, Phenotypes, Disorders, Signaling Molecules, Events, Medical Procedures, Activities, Functions, and Money. Output the recognized entities in JSON format, with each entity represented as {\"entity_text\": \"text\", \"entity_label\": \"category\"}. Note: 1. Each entity must be output as a correct JSON string. 2. If no entities are found, output \"No entities found\".",
        "input": f"Text: {sentence}",
        "output": format_entities(sentence, entities)
    }

    formatted_data.append(formatted_entry)

# 将格式化后的数据写入新的 JSON 文件
output_file = 'formatted_biomed_ner_data.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(formatted_data, f, ensure_ascii=False, indent=4)

print(f"Data has been formatted and saved to {output_file}")
