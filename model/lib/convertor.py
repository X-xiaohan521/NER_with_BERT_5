import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel, BertTokenizer, BertConfig
import re

def label_dictionary(choose):
    '''
    数字编码字典。 \n
    参数：choose(str):如果为"labels_to_int"，则返回标签对编码字典；如果为"int_to_labels"，则返回编码对标签字典；如果为"num_labels"，则返回标签数量，否则返回None   \n
    编码字典：\n
    0 - CLS
    0 - SEP
    0 - PAD

    1 - B-PER \n
    2 - I-PER \n
    3 - B-LOC \n
    4 - I-LOC \n
    5 - B-ORG \n

    6 - I-ORG \n
    7 - B-DATE \n
    8 - I-DATE \n
    9 - O
    '''
    #定义标签to数字编码字典
    labels_to_int = {"B-PER":1, "I-PER":2, "B-LOC":3, "I-LOC":4, "B-ORG":5, "I-ORG":6, "B-DATE":7, "I-DATE":8, "O":9}
    int_to_labels = {value: key for key, value in labels_to_int.items()}

    if choose == "labels_to_int":
        return labels_to_int
    elif choose == "int_to_labels":
        return int_to_labels
    elif choose == "num_labels":
        return len(labels_to_int)+3
    else:
        return None

class MyTokenizer:
    # 自定义分词器，将数字分开
    def __init__(self, model_path):
        self.tokenizer = BertTokenizer.from_pretrained(model_path, encoding="utf-8")

    def tokenize(self, text):
        # 使用正则表达式将数字分开
        text = re.sub(r'(\d)', r' \1 ', text)
        text = re.sub(r'(\w)', r' \1 ', text)

        # 添加开始标签[CLS]
        text = "[CLS] " + text
        # 添加结束标签[SEP]
        text = text + " [SEP]"

        tokens = self.tokenizer.tokenize(text)

        # 将分词后的结果转换为对应的标记ID
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # 生成attention_mask，标记部分为1，padding部分为0
        attention_mask = [1] * len(input_ids)

        return {"input_ids": torch.tensor([input_ids]), "attention_mask": torch.tensor([attention_mask])}


def data_preprocess(model_path, data_path):
    '''
    数据预处理函数，用于将数据集转换为BERT可接受的数据格式。\n
    参数：
    model_path(str):预训练BERT模型路径 \n
    data_path(str):待处理数据集路径 \n
    返回：
    一个列表，其中每个元素为一个字典，每个字典包含三对键值，键："input_ids", "attention_mask", "labels"，每一个值都是张量
    '''
    temp_sentence=""
    sentences_list=[]  #一个列表，列表中每个元素都是一个字符串，其中包含一句句子
    #读取文件
    with open(data_path,"r",encoding="utf-8") as f:
        #拆分数据集为一个列表，列表中每个元素都是一个字符串，其中包含一句句子
        for line in f:
            if line=="\n":
                sentences_list.append(temp_sentence)
                temp_sentence=""
            else:
                temp_sentence+=line[0]
    
    #定义与处理过的数据列表，其中每个元素为一个字典，每个字典包含两对键值，键："input_ids", "attention_mask"，每一个值都是张量
    preprocessed_data=[]
    tokenizer=MyTokenizer(model_path)
    for sentence in sentences_list:
        preprocessed_data.append(tokenizer.tokenize(sentence))

    #调用获取labels方法，返回一个列表，列表中每个元素为一个小列表，小列表中每个元素为一个int，代表每个token对应的标签
    label_list=get_labels(data_path)

    for i in range(len(preprocessed_data)):
        #将标签添加进preprocessed_data中
        preprocessed_data[i]["labels"]=torch.tensor([label_list[i]])
    return preprocessed_data


def get_labels(data_path):
    '''
    输入数据集，提取标签并数字化。 \n
    参数：
    data_path(str):待处理数据集路径 \n
    返回：
    一个列表，列表中每个元素为一个小列表，小列表中每个元素为一个int，代表每个token对应的标签\n
    '''
    
    #定义列表
    temp_label_list=[]  #临时标签列表
    label_list=[]  #最后返回列表

    #打开文件
    with open(data_path,"r",encoding="utf-8") as f:
        for line in f:
            if line=="\n":
                #补齐首尾
                temp_label_list.insert(0,0)
                temp_label_list.append(0)

                label_list.append(temp_label_list)
                temp_label_list=[]
            else:
                temp_label_list.append(label_dictionary("labels_to_int")[line[2:].replace("\n","")])
    return label_list
