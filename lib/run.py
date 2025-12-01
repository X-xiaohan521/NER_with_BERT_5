import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
import os
import lib.convertor as convertor
import lib.model_evaluator as model_evaluator
from lib.NERModel import NERModel
from lib.checkpoints import load_weights


#判断所使用的设备
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
torch.device(device)

def test_model(model_path='./weight/ner/model.pth',
                data_path='./data/newtest.txt', 
                pretrained_model_path='./weight/chinese_rbtl3_pytorch/'):

    #读取标签种类数
    num_labels = convertor.label_dictionary("num_labels")


    #导入模型
    #导入BERT的超参数
    config = BertConfig.from_json_file(os.path.join(pretrained_model_path, 'bert_config.json'))

    #根据BERT的超参数导入BERT模型权重
    bert_model = BertModel.from_pretrained(pretrained_model_path, config=config)

    #定义NER模型架构
    ner_model = NERModel(bert_model, config, num_labels)
    ner_model = ner_model.to(device)

    #导入NER模型权重
    load_weights(ner_model, model_path, strict=False)


    #调用数据预处理方法，将测试集转换为模型接受的输入
    test_data = convertor.data_preprocess(pretrained_model_path, data_path)


    #定义损失函数
    criterion = nn.CrossEntropyLoss()


    #调用模型评估函数
    result = model_evaluator.evaluate_model(ner_model,test_data,criterion,num_labels)

    return result