import torch
from transformers import BertModel, BertTokenizer, BertConfig
import torch.nn as nn
import torch.optim as optim
import logging
import lib.convertor as convertor
import lib.model_evaluator as model_evaluator
from lib.NERModel import NERModel


def train(pretrained_model_path = './weight/chinese_rbtl3_pytorch/',
           data_path = './data/train_1.txt', 
           test_path = './data/newtest.txt',
            output_model_path = './weight/ner/', 
            num_epochs = 1):
    
    #配置日志设置
    logging.basicConfig(
        level=logging.INFO,  # 将日志级别设置为INFO
        format='%(asctime)s [%(levelname)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    #判断所使用的设备
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    torch.device(device)

    #加载本地预训练的BERT模型超参数
    config = BertConfig.from_json_file(pretrained_model_path + 'bert_config.json')

    #根据超参数加载本地预训练的BERT模型权重
    bert_model = BertModel.from_pretrained(pretrained_model_path, config=config)

    #加载单词表（用于反分词器）
    detokenizer = BertTokenizer.from_pretrained(pretrained_model_path)

    #调用数据预处理方法，返回：一个列表，其中每个元素为一个字典，每个字典包含三对键值，键："input_ids", "attention_mask", "labels"，每一个值都是张量
    training_data=convertor.data_preprocess(pretrained_model_path, data_path)

    #初始化NER模型
    num_labels = convertor.label_dictionary("num_labels")  #获取标签数量
    ner_model = NERModel(bert_model, config, num_labels)
    ner_model = ner_model.to(device)

    #定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ner_model.parameters(), lr=1e-5)

    #训练模型
    total_steps = len(training_data) * num_epochs  #计算总步骤（即训练集中总共有几句话）
    for epoch in range(num_epochs):
        # 用训练数据集迭代训练模型
        for step, dict in enumerate(training_data, 1):  # 添加步骤计数器（从1开始）
            #遍历training_data列表中的每一个字典
            optimizer.zero_grad()  #优化器梯度归零
            #从字典中分别取出键input_ids和attention_mask对应的值（这些值均为张量）
            input_ids=dict["input_ids"]
            input_ids = input_ids.to(device)
            attention_mask=dict["attention_mask"]
            attention_mask = attention_mask.to(device)
            print(detokenizer.decode(input_ids[0]))
            try:
                #尝试向前传递input_ids,若失败，则跳过当前step继续运行，并报错
                logits = ner_model.forward(input_ids, attention_mask)
            
                #计算loss
                loss = criterion(logits.view(-1, num_labels), dict["labels"].to(device).view(-1))

                #记录训练进度
                avg_loss = loss.item() / step   #计算每步的平均损失
                logging.info(f"Epoch [{epoch + 1}/{num_epochs}] | Step [{step}/{len(training_data)}] | Loss: {avg_loss:.8f}")

                #反向更新与梯度下降
                loss.backward()
                optimizer.step()
            except:
                optimizer.zero_grad()  #优化器梯度归零
                logging.error(f"Step [{step}/{len(training_data)}] | Unknown token")
                continue

    #评估模型
    #预处理验证集数据
    test_data=convertor.data_preprocess(pretrained_model_path, test_path)

    #调用模型评估函数
    model_evaluator.evaluate_model(ner_model,test_data,criterion,num_labels)

    #保存模型
    torch.save(ner_model.state_dict(), output_model_path + 'ner_model.pth')
    logging.info(f"Model Saved")

    return None