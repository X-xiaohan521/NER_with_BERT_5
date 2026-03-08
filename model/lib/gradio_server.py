import gradio as gr
from lib.user_interact import user_interact

#设置路径
model_path = './weight/ner/model_new.pth'  #设置模型路径
pretrained_model_path = './weight/chinese_rbtl3_pytorch/'  #设置预训练BERT模型路径

#创建ui对象并初始化
ui = user_interact(model_path, pretrained_model_path)

#调用gradio库创建UI界面
user_interface = gr.Interface(fn=ui.user_interact, 
                              inputs=gr.Textbox(lines=2, label="输入文本", placeholder="请在此输入用于命名实体识别的句子……"), 
                              outputs=gr.HighlightedText(label="结果", color_map={"B-PER":"#16A34A", "I-PER":"#16A34A", "B-LOC":"#DC2626", "I-LOC":"#DC2626", "B-ORG":"#9333EA", "I-ORG":"#9333EA", "B-DATE":"#EAB308", "I-DATE":"#EAB308"}),
                              title="命名实体识别")

#启动UI服务器
user_interface.launch()