
import pandas as pd
import numpy as np
import random
import streamlit as st
from streamlit.logger import get_logger
from tqdm import tqdm
import numpy as np
import transformers
transformers.logging.set_verbosity_error()
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import BertModel, BertConfig
import os
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import Dataset


LOGGER = get_logger(__name__)
def number_to_letter(number):
    # 创建一个字典，将数字映射到对应的字母
    mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    # 通过字典获取对应的字母
    return mapping.get(number, "Invalid number")  # 如果数字不在0到3之间，返回错误信息

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MyDataset(Dataset):
    def __init__(self, dataframe):
        # 如果传进来的dataframe其实是单个样本，将其转化为DataFrame
        self.df = dataframe if isinstance(dataframe, pd.DataFrame) else pd.DataFrame([dataframe])


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):  # 将一条数据从(文章,问题,4个选项)转成(文章,问题,选项1)、(文章,问题,选项2)...
#         label = self.df.label.values[idx]
        label = self.df.iloc[idx].label  # 使用iloc来兼容单个样本的情况
        question = self.df.Question.values[idx]
        content = self.df.Content.values[idx]
        choice = self.df.Choices.values[idx][2:-2].split('\', \'')
        if len(choice) < 4:  # 如果选项不满四个，就补“不知道”
            for i in range(4 - len(choice)):
                choice.append('D．不知道')

        content = [content for i in range(len(choice))]
        pair = [question + ' ' + i[2:] for i in choice]

        return content, pair, label

def run():
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

    st.write("# Welcome here! 👋")

    st.sidebar.success("Select a demo above.")

    test_df = pd.read_csv('/workspaces/gra002/test.csv')
    # 获取DataFrame的行数
    num_rows = len(test_df)

    if st.button('给我一个问题'):
        st.session_state.index = random.randint(0, num_rows - 1)
        test_df = pd.read_csv('/workspaces/gra002/test.csv')
        test_df['label'] = 0

        CFG = {  # 训练的参数配置
            'fold_num': 5,  # 五折交叉验证
            'seed': 42,
            'model': 'hfl/chinese-macbert-large',  # 预训练模型
            'max_len': 256,  # 文本截断的最大长度
            # 'epochs': 12,
            'epochs': 10,
            'train_bs': 4,  # batch_size，可根据自己的显存调整
            'valid_bs': 4,
            'lr': 2e-5,  # 学习率
            'lrSelf': 1e-4,  # 学习率
            'num_workers': 8,
            'accum_iter': 8,  # 梯度累积，相当于将batch_size*2
            'weight_decay': 1e-4,  # 权重衰减，防止过拟合
            'device': 0,
            'adv_lr': 0.01,
            'adv_norm_type': 'l2',
            'adv_init_mag': 0.03,
            'adv_max_norm': 1.0,
            'ip': 2,
            'gpuNum': 1
        }

        # 假设你知道你想要加载模型的确切类型和名称
        model_name = 'hfl/chinese-macbert-large'

        # 首先从预训练模型名加载配置
        config = BertConfig.from_pretrained(model_name)

        # 通过配置实例化模型
        model = BertModel(config)

        # 确保模型处于评估模式
        model.eval()

        # 现在模型已经加载并准备好进行预测或其他操作
        tokenizer = BertTokenizer.from_pretrained(CFG['model'])  # 加载bert的分词器
        def collate_fn(data):  # 将文章问题选项拼在一起后，得到分词后的数字id，输出的size是(batch, n_choices, max_len)
            input_ids, attention_mask, token_type_ids = [], [], []
            for x in data:
                text = tokenizer(x[1], text_pair=x[0], padding='max_length', truncation=True, max_length=CFG['max_len'],
                                return_tensors='pt')
                input_ids.append(text['input_ids'].tolist())
                attention_mask.append(text['attention_mask'].tolist())
                token_type_ids.append(text['token_type_ids'].tolist())
            input_ids = torch.tensor(input_ids)
            attention_mask = torch.tensor(attention_mask)
            token_type_ids = torch.tensor(token_type_ids)
            label = torch.tensor([x[-1] for x in data])
            return input_ids, attention_mask, token_type_ids, label

        test_set = test_df.iloc[[st.session_state.index]]  # 注意双括号，这样返回的是DataFrame而不是Series

        test_set = MyDataset(test_set)
        test_loader = DataLoader(test_set, batch_size=1, collate_fn=collate_fn, shuffle=False,
                                num_workers=1)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        y_pred = []
        predictions = []

        with torch.no_grad():
            tk = tqdm(test_loader, total=len(test_loader), position=0, leave=True)
            for idx, (input_ids, attention_mask, token_type_ids, y) in enumerate(tk):
                input_ids, attention_mask, token_type_ids, y = input_ids.to(device), attention_mask.to(
                    device), token_type_ids.to(device), y.to(device).long()
            
                input_ids = input_ids.squeeze(0)
                attention_mask = attention_mask.squeeze(0)
                token_type_ids = token_type_ids.squeeze(0)            
                
                output = model(input_ids, attention_mask, token_type_ids)[0].cpu ().numpy()
                token_scores = np.max(output, axis=2)

                option_scores = np.mean(token_scores, axis=1)

                predicted_option = np.argmax(option_scores)
                st.session_state.pre=predicted_option
        # print(option_scores)

        prediction=number_to_letter(predicted_option)
        print(prediction)    
        
    st.write(test_df.iloc[st.session_state.index])

    genre = st.radio(
        "你的答案是：",
        ["A ::smiling_face_with_3_hearts:", "B ::thermometer:", "C :blush:","D ::sun_with_face:"],
        # captions = ["Laugh out loud.", "Get the popcorn.", "Never stop learning."]
        index=None,)


    if(st.session_state.pre==0):
        if genre == 'A ::smiling_face_with_3_hearts:':
            st.balloons()
            st.write('You selected right!')
        else:
            st.write("You  select wrong!.")       

    elif(st.session_state.pre==1):
        if genre == 'B ::thermometer:':
            st.balloons()
            st.write('You selected right!')
        else:
            st.write("You  select wrong!.")  

    elif(st.session_state.pre==2):
        if genre == 'C :blush:':
            st.balloons()
            st.write('You selected right!')
        else:
            st.write("You  select wrong!.")  

    elif(st.session_state.pre==3):
        if genre == 'D ::sun_with_face:':
            st.balloons()
            st.write('You selected right!')
        else:
            st.write("You  select wrong!.")  


    st.markdown(
        """
        本项目的数据来自小学/中高考语文阅读理解题库。相较于英文，中文阅读理解有着更多的歧义性和多义性，然而璀璨的中华文明得以绵延数千年，离不开每一个时代里努力钻研、坚守传承的人，这也正是本项目的魅力与挑战，让机器读懂文字，让机器学习文明。

        **☝ 随机生成一道题目来测试一下吧** 

    """
    )


if __name__ == "__main__":
    if "index" not in st.session_state:
        st.session_state.index = 0

    if "pre" not in st.session_state:
        st.session_state.pre = 0
    run()

