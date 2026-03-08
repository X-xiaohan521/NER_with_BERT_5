import torch

#判断所使用的设备
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
torch.device(device)

def evaluate_model(model, evaluation_data, criterion, num_labels):
    '''
    模型评估函数，用于评估训练好的模型。 \n
    参数： \n
    model: 待评估模型（要求已经完成实例化） \n
    evaluation_data: 测试集（要求一个列表，其中每个元素为一个字典，每个字典包含三对键值，键："input_ids", "attention_mask", "labels"，每一个值都是张量）  \n
    criterion: 损失函数（要求已经完成实例化）  \n
    num_labels(int): NER任务的标签种类数量  \n
    返回：一个元组，包含四个元素，分别是：  \n
    total_tokens(int): 测试集数据总数  \n
    avg_loss(int): 平均loss  \n
    accuracy(int): 准确率  \n
    recall(int): 召回率  \n
    f1_score(int): F1分数  \n
    '''
    model.eval()  # 设置模型为评估模式

    total_tokens = 0
    total_correct = 0
    total_predicted = 0
    total_actual = 0
    total_loss = 0.0

    with torch.no_grad():
        for dict in evaluation_data:
            input_ids = dict["input_ids"]
            input_ids = input_ids.to(device)

            attention_mask = dict["attention_mask"]
            attention_mask = attention_mask.to(device)

            labels = dict["labels"]
            labels = labels.to(device)
            
            try:
                logits = model(input_ids, attention_mask)
                loss = criterion(logits.view(-1, num_labels), labels.view(-1))
                total_loss += loss.item()

                predicted_labels = torch.argmax(logits, dim=-1)

                total_tokens += input_ids.size(0)
                total_correct += torch.sum(predicted_labels == labels).item()
                total_predicted += torch.sum(predicted_labels != 0).item()  # Count non-padding tokens
                total_actual += torch.sum(labels != 0).item()  # Count non-padding tokens
            except:
                continue

    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_actual
    recall = total_correct / total_predicted
    f1_score = 2 * (accuracy * recall) / (accuracy + recall)

    print(f"Total Samples: {total_tokens}, Eval Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}")

    return (total_tokens,avg_loss, accuracy, recall, f1_score) 
