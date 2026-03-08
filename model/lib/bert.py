import torch
from transformers import BertModel, BertTokenizer, BertConfig

# 定义本地预训练模型的路径
model_path = '/home/ma-user/work/BERT/NER_with_BERT/model/pretrained_models/chinese_rbtl3_pytorch/'

# 加载本地预训练的BERT模型配置
config = BertConfig.from_json_file(model_path + 'bert_config.json')

# 加载本地预训练的BERT模型和分词器
bert_model = BertModel.from_pretrained(model_path, config=config)
tokenizer = BertTokenizer.from_pretrained(model_path)

# 输入示例文本
text = "你好，很高兴遇见你。你好，我也很高兴，请多指教。"

# 使用分词器对文本进行编码
inputs = tokenizer(text, return_tensors="pt")

# 将编码后的文本输入BERT模型
outputs = bert_model(**inputs)

# 输出BERT模型的结果
print(outputs.last_hidden_state)

