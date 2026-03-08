import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
import os
from lib.NERModel import NERModel
import lib.convertor as convertor
from lib.checkpoints import load_weights

#判断所使用的设备
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
torch.device(device)

class user_interact():
    def __init__(self, model_path='./weight/ner/model.pth', pretrained_model_path='./weight/chinese_rbtl3_pytorch/'):
        
        #将外部传入的参数作为类的对象的属性存储
        self.model_path = model_path
        self.pretrained_model_path = pretrained_model_path

        #读取标签数量
        self.num_labels = convertor.label_dictionary("num_labels")

        #读取编码对标签字典
        self.int_to_labels = convertor.label_dictionary("int_to_labels")

        #导入模型
        #导入BERT的超参数
        self.config = BertConfig.from_json_file(os.path.join(self.pretrained_model_path, 'bert_config.json'))

        #根据BERT的超参数导入BERT模型权重
        self.bert_model = BertModel.from_pretrained(self.pretrained_model_path, config=self.config)

        #定义NER模型架构
        self.ner_model = NERModel(self.bert_model, self.config, self.num_labels)
        self.ner_model = self.ner_model.to(device)

        #导入NER模型权重
        load_weights(self.ner_model, self.model_path, strict=False)

        #设置模型为评估模式
        self.ner_model.eval()

    def user_interact(self, user_input):

        #数据预处理
        #创建分词器对象
        tokenizer = convertor.MyTokenizer(self.pretrained_model_path)

        #调用tokenize方法，将用户输入的句子转换为模型可接受的格式
        input_data = tokenizer.tokenize(user_input)

        #向前传播
        with torch.no_grad():
            input_ids = input_data["input_ids"]
            attention_mask = input_data["attention_mask"]
            logits = self.ner_model(input_ids, attention_mask)
            predicted_labels = torch.argmax(logits, dim=-1)

        #编码转换
        #tensor转list
        predicted_labels_list = predicted_labels.tolist()

        #二维转一维
        predicted_labels_list = predicted_labels_list[0]

        #定义临时列表
        temp = []

        #数字转str
        for i in predicted_labels_list:
            try:
                temp.append(self.int_to_labels[i])
            except:
                continue
        
        #把"O"转换成None
        for i in range(len(temp)):
            if temp[i] == "O":
                temp[i] = None

        #定义输出列表
        output=[]

        for i in range(len(user_input)):
            output.append((user_input[i], temp[i]))

        #返回结果
        return output