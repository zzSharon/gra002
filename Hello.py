
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

import streamlit as st

# # 初始化 session_state 里的 users
# if 'users' not in st.session_state:
#     st.session_state['xkp1234'] = {}

# if 'login_status' not in st.session_state:
#     st.session_state['login_status'] = False

# if 'current_user' not in st.session_state:
#     st.session_state['current_user'] = ''

# def login_user(username, password):
#     """校验用户登录信息"""
#     return st.session_state.get(username) == password

# def main_page():
#     # st.write(f"欢迎 {st.session_state['current_user']}, 你已成功登录！")
#     # 这里添加登录成功后显示的页面内容


#     st.write("# 您已成功登录😊！")
#     st.sidebar.success("这里是答题页面，生成问题后查看预测答案.")

#     test_df = pd.read_csv('/workspaces/gra002/test.csv')
#     # 获取DataFrame的行数
#     num_rows = len(test_df)

#     if st.button('给我一个问题'):
#         st.session_state.index = random.randint(0, num_rows - 1)
#         test_df = pd.read_csv('/workspaces/gra002/test.csv')
#         test_df['label'] = 0

#         CFG = {  # 训练的参数配置
#             'fold_num': 5,  # 五折交叉验证
#             'seed': 42,
#             'model': 'hfl/chinese-macbert-large',  # 预训练模型
#             'max_len': 256,  # 文本截断的最大长度
#             # 'epochs': 12,
#             'epochs': 10,
#             'train_bs': 4,  # batch_size，可根据自己的显存调整
#             'valid_bs': 4,
#             'lr': 2e-5,  # 学习率
#             'lrSelf': 1e-4,  # 学习率
#             'num_workers': 8,
#             'accum_iter': 8,  # 梯度累积，相当于将batch_size*2
#             'weight_decay': 1e-4,  # 权重衰减，防止过拟合
#             'device': 0,
#             'adv_lr': 0.01,
#             'adv_norm_type': 'l2',
#             'adv_init_mag': 0.03,
#             'adv_max_norm': 1.0,
#             'ip': 2,
#             'gpuNum': 1
#         }
#         st.markdown("***")
#         # 假设你知道你想要加载模型的确切类型和名称
#         model_name = 'hfl/chinese-macbert-large'

#         # 首先从预训练模型名加载配置
#         config = BertConfig.from_pretrained(model_name)

#         # 通过配置实例化模型
#         model = BertModel(config)

#         # 确保模型处于评估模式
#         model.eval()

#         # 现在模型已经加载并准备好进行预测或其他操作
#         tokenizer = BertTokenizer.from_pretrained(CFG['model'])  # 加载bert的分词器
#         def collate_fn(data):  # 将文章问题选项拼在一起后，得到分词后的数字id，输出的size是(batch, n_choices, max_len)
#             input_ids, attention_mask, token_type_ids = [], [], []
#             for x in data:
#                 text = tokenizer(x[1], text_pair=x[0], padding='max_length', truncation=True, max_length=CFG['max_len'],
#                                 return_tensors='pt')
#                 input_ids.append(text['input_ids'].tolist())
#                 attention_mask.append(text['attention_mask'].tolist())
#                 token_type_ids.append(text['token_type_ids'].tolist())
#             input_ids = torch.tensor(input_ids)
#             attention_mask = torch.tensor(attention_mask)
#             token_type_ids = torch.tensor(token_type_ids)
#             label = torch.tensor([x[-1] for x in data])
#             return input_ids, attention_mask, token_type_ids, label

#         test_set = test_df.iloc[[st.session_state.index]]  # 注意双括号，这样返回的是DataFrame而不是Series

#         test_set = MyDataset(test_set)
#         test_loader = DataLoader(test_set, batch_size=1, collate_fn=collate_fn, shuffle=False,
#                                 num_workers=1)

#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         model.to(device)
#         y_pred = []
#         predictions = []

#         with torch.no_grad():
#             tk = tqdm(test_loader, total=len(test_loader), position=0, leave=True)
#             for idx, (input_ids, attention_mask, token_type_ids, y) in enumerate(tk):
#                 input_ids, attention_mask, token_type_ids, y = input_ids.to(device), attention_mask.to(
#                     device), token_type_ids.to(device), y.to(device).long()
            
#                 input_ids = input_ids.squeeze(0)
#                 attention_mask = attention_mask.squeeze(0)
#                 token_type_ids = token_type_ids.squeeze(0)            
                
#                 output = model(input_ids, attention_mask, token_type_ids)[0].cpu ().numpy()
#                 token_scores = np.max(output, axis=2)

#                 option_scores = np.mean(token_scores, axis=1)

#                 predicted_option = np.argmax(option_scores)
#                 st.session_state.pre=predicted_option
#         # print(option_scores)

#         prediction=number_to_letter(predicted_option)
#         print(prediction)    

#     # st.write(test_df.iloc[st.session_state.index])
#     st.markdown("***文章:***")
#     st.write("   ",test_df.iloc[st.session_state.index].Content)

#     st.markdown("---")


#     st.markdown("***问题:***")
#     st.write(test_df.iloc[st.session_state.index].Question,":")
#     st.markdown("---")
#     # st.write(test_df.iloc[st.session_state.index].Choices)
    
#     choices=test_df.iloc[st.session_state.index].Choices
#     # 初始化一个空列表来存储提取后的选项
#     extracted_choices = []

#     # 初始化一个变量来跟踪当前正在构建的选项
#     current_option = ''

#     # 遍历 choices 列表中的每个元素
#     for element in choices:
#         # 如果元素是字母 'A' 或 'B'，则将其添加到当前选项中
#         if element in ['A', 'B','C','D']:
#             # 如果当前选项不为空，则将其添加到提取后的选项列表中，并重置当前选项
#             if current_option:
#                 extracted_choices.append(current_option)
#                 current_option = ''
#             current_option += element
#         # 如果元素是中文或其他字符，则将其添加到当前选项中
#         elif element not in ['[', ']',',',"'"]:
#             current_option += element

#     # 添加最后一个选项到列表中
#     if current_option:
#         extracted_choices.append(current_option)

#     print(extracted_choices[0])

#     st.markdown("***选项:***")
#     choice_a=extracted_choices[0]
#     choice_b=extracted_choices[1]
#     choice_c=extracted_choices[2]
#     choice_d=extracted_choices[3]

#     st.write(choice_a)
#     st.write(choice_b)
#     st.write(choice_c)
#     st.write(choice_d)
#     st.markdown("---")

#     st.markdown("***答案:***")
#     if(st.session_state.pre==0):
#         st.write(choice_a)
     
#     elif(st.session_state.pre==1):
#         st.write(choice_b)

#     elif(st.session_state.pre==2):
#         st.write(choice_c)

#     elif(st.session_state.pre==3):
#         st.write(choice_d)
#     st.markdown("---")

#     st.markdown(
#         """
#         本项目的数据来自小学/中高考语文阅读理解题库。相较于英文，中文阅读理解有着更多的歧义性和多义性，然而璀璨的中华文明得以绵延数千年，离不开每一个时代里努力钻研、坚守传承的人。

#         **☝ 随机生成一道题目来测试一下吧** 

#     """
#     )


# import streamlit as st

# def login_page():
#     st.title("欢迎来到中学阅读理解系统！🥰")
#     st.markdown(
#         """
#         系统模型基于NCR数据集训练，NCR数据集是一个专门为机器阅读理解（MRC）设计的综合性中文数据集，它包含有8000余份文本，平均长度为1040个字符，远远超过现有中文MRC数据集的平均长度，处理这样的长文本会面临不小的问题。这些文本涵盖了广泛的中国写作风格，包括文言文、现代文章、古典诗歌等，大部分中文数据集都没有涉及这么多样的题材，且题目难度较大，对母语为汉语的人都有一定难度，这为机器提供了更加丰富和多样化的阅读材料；NCR数据集中含有两万多个问题，这些问题不仅数目众多而且需要很强的推理能力和常识才能找到正确答案，这对汉语母语者也有相当的难度。

#     """
#     )
#     st.markdown(
#     " ***👇 下面是关于训练集、数据集的文章长度***."
#     )
#     st.image('train_len.png')
#     st.image('test_len.png')
#     st.markdown(
#     " ***👇 文章个数统计***."
#     )
#     st.image('answer_count.png')
#     st.markdown(
#     " ***👈 点击左边侧边栏，登录后进入系统***."
#     )

#     # 创建一个选择器让用户选择登录或注册
#     choice = st.sidebar.selectbox("登录或注册", ["登录", "注册"])

#     if choice == "登录":
#         username = st.sidebar.text_input("用户名")
#         password = st.sidebar.text_input("密码", type='password')
#         if st.sidebar.button("登录"):
#             if login_user(username, password):
#                 st.success("登录成功!")
#                 st.session_state['login_status'] = True
#                 st.session_state['current_user'] = username
#             else:
#                 st.error("用户名或密码错误")

#     elif choice == "注册":
#         new_username = st.sidebar.text_input("选择一个用户名", key="new_username")
#         new_password = st.sidebar.text_input("设置一个密码", type='password', key="new_password")
#         confirm_password = st.sidebar.text_input("确认密码", type='password', key="confirm_password")
#         if st.sidebar.button("注册"):
#             if new_password == confirm_password:
#                 # 在 session_state 中存储新用户
#                 st.session_state[new_username] = {}
#                 st.session_state[new_username] = new_password
#                 st.success(f"用户 {new_username} 注册成功!")
#             else:
#                 st.error("两次输入的密码不匹配")

# if __name__ == "__main__":
#     if st.session_state['login_status']:
#         main_page()
#     else:
#         login_page()
#     if "index" not in st.session_state:
#         st.session_state.index = 0

#     if "pre" not in st.session_state:
#         st.session_state.pre = 0

# 初始化 session_state 里的 users
if 'users' not in st.session_state:
    st.session_state['xkp1234'] = {}

if 'login_status' not in st.session_state:
    st.session_state['login_status'] = False

if 'current_user' not in st.session_state:
    st.session_state['current_user'] = ''

def login_user(username, password):
    """校验用户登录信息"""
    return st.session_state.get(username) == password

def main_page():
    # st.write(f"欢迎 {st.session_state['current_user']}, 你已成功登录！")
    # 这里添加登录成功后显示的页面内容


    st.write("# 您已成功登录😊！")
    st.sidebar.success("这里是答题页面，生成问题后查看预测答案.")

    test_df = pd.read_csv('/workspaces/gra002/test.csv')
    test_df['label'] = 0

    # 获取DataFrame的行数
    num_rows = len(test_df)

    # 定义选项列表
    options = ["11", "0", "22", "33"]
    options = ["文言文", "现代文", "古诗", "现代诗"]
    # 使用st.selectbox创建选项栏
    selected_option = st.selectbox("请从以下选项中选择一项：", options)

    # 显示用户选择的选项
    st.write(f"您选择了：{selected_option}")

    if selected_option == "文言文":
        test_df = test_df[test_df['Type'] == 11]
        num_rows = len(test_df)
    if selected_option == "现代文":
        test_df = test_df[test_df['Type'] == 0]
        num_rows = len(test_df)
    if selected_option == "古诗":
        test_df = test_df[test_df['Type'] == 22]
        num_rows = len(test_df)
    if selected_option == "现代诗":
        test_df = test_df[test_df['Type'] == 33]
        num_rows = len(test_df)

    unique_types = test_df['Diff'].unique()

    # 使用 st.selectbox 创建一个下拉选择框
    selected_value = st.selectbox("请从以下数值中选择一个：", unique_types)
    test_df = test_df[test_df['Diff'] == selected_value]
    num_rows = len(test_df)

    # 创建一个下拉选择框，选项是从 0 到 Len
    selected_option = st.selectbox("请选择题号", list(range(num_rows )))

    # 显示用户选择的数字
    st.write("你选择的题号是:", selected_option)

    if st.button('生成问题'):
        st.session_state.index = selected_option

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
        st.markdown("***")
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

    # st.write(test_df.iloc[st.session_state.index])
    st.markdown("***文章:***")
    st.write("   ",test_df.iloc[st.session_state.index].Content)

    st.markdown("---")


    st.markdown("***问题:***")
    st.write(test_df.iloc[st.session_state.index].Question,":")
    st.markdown("---")
    # st.write(test_df.iloc[st.session_state.index].Choices)
    
    choices=test_df.iloc[st.session_state.index].Choices
    # 初始化一个空列表来存储提取后的选项
    extracted_choices = []

    # 初始化一个变量来跟踪当前正在构建的选项
    current_option = ''

    # 遍历 choices 列表中的每个元素
    for element in choices:
        # 如果元素是字母 'A' 或 'B'，则将其添加到当前选项中
        if element in ['A', 'B','C','D']:
            # 如果当前选项不为空，则将其添加到提取后的选项列表中，并重置当前选项
            if current_option:
                extracted_choices.append(current_option)
                current_option = ''
            current_option += element
        # 如果元素是中文或其他字符，则将其添加到当前选项中
        elif element not in ['[', ']',',',"'"]:
            current_option += element

    # 添加最后一个选项到列表中
    if current_option:
        extracted_choices.append(current_option)

    print(extracted_choices[0])

    st.markdown("***选项:***")
    choice_a=extracted_choices[0]
    choice_b=extracted_choices[1]
    choice_c=extracted_choices[2]
    choice_d=extracted_choices[3]

    st.write(choice_a)
    st.write(choice_b)
    st.write(choice_c)
    st.write(choice_d)
    st.markdown("---")



    # st.markdown("***答案:***")

    # 创建一个按钮
    if st.button('生成答案'):
        # 当按钮被点击时，显示以下消息
        if(st.session_state.pre==0):
            st.write(choice_a)
        
        elif(st.session_state.pre==1):
            st.write(choice_b)

        elif(st.session_state.pre==2):
            st.write(choice_c)

        elif(st.session_state.pre==3):
            st.write(choice_d)
    # if(st.session_state.pre==0):
    #     st.write(choice_a)
     
    # elif(st.session_state.pre==1):
    #     st.write(choice_b)

    # elif(st.session_state.pre==2):
    #     st.write(choice_c)

    # elif(st.session_state.pre==3):
    #     st.write(choice_d)
    st.markdown("---")

    st.markdown(
        """
        本项目的数据来自小学/中高考语文阅读理解题库。相较于英文，中文阅读理解有着更多的歧义性和多义性，然而璀璨的中华文明得以绵延数千年，离不开每一个时代里努力钻研、坚守传承的人。

        **☝ 随机生成一道题目来测试一下吧** 

    """
    )


# CSV 文件路径
users_file_path = 'users.csv'

# 确保 CSV 文件存在，如果不存在则创建
if not os.path.isfile(users_file_path):
    # 创建一个空的 DataFrame，并添加列名
    df_empty = pd.DataFrame(columns=['username', 'password'])
    # 将空 DataFrame 保存到 CSV 文件
    df_empty.to_csv(users_file_path, index=False)

def load_users():
    """从 CSV 文件加载用户数据"""
    return pd.read_csv(users_file_path)

def save_user(username, password):
    """保存新用户到 CSV 文件"""
    df_users = load_users()
    new_user_df = pd.DataFrame([[username, password]], columns=['username', 'password'])
    df_users = pd.concat([df_users, new_user_df], ignore_index=True)
    df_users.to_csv(users_file_path, index=False)

def user_exists(username, df_users):
    """检查用户名是否存在"""
    return df_users['username'].str.lower().str.strip().eq(username.lower().strip()).any()

def check_user(username, password, df_users):
    # 确保username和password列是字符串类型
    df_users[['username', 'password']] = df_users[['username', 'password']].astype(str)

    # 去除输入和DataFrame中的空白字符，并进行大小写不敏感比较
    username = username.strip().lower()
    password = password.strip()
    
    # 使用查询的方式来检查用户名和密码
    user = df_users.query('username.str.lower().str.strip() == @username and password.str.strip() == @password', engine='python')
    
    # 直接返回检查结果
    return not user.empty

# Streamlit 应用逻辑


def login_page():
    st.sidebar.markdown("## 登录或注册")
    st.title("欢迎来到中学阅读理解系统！🥰")
    st.markdown(
        """
        系统模型基于NCR数据集训练，NCR数据集是一个专门为机器阅读理解（MRC）设计的综合性中文数据集，它包含有8000余份文本，平均长度为1040个字符，远远超过现有中文MRC数据集的平均长度，处理这样的长文本会面临不小的问题。这些文本涵盖了广泛的中国写作风格，包括文言文、现代文章、古典诗歌等，大部分中文数据集都没有涉及这么多样的题材，且题目难度较大，对母语为汉语的人都有一定难度，这为机器提供了更加丰富和多样化的阅读材料；NCR数据集中含有两万多个问题，这些问题不仅数目众多而且需要很强的推理能力和常识才能找到正确答案，这对汉语母语者也有相当的难度。

    """
    )
    st.markdown(
    " *** 下面是关于训练集、数据集的文章长度***."
    )
    st.image('train_len.png')
    st.image('test_len.png')
    st.markdown(
    " *** 文章个数统计***."
    )
    st.image('answer_count.png')
    st.markdown(
    " *** 点击左边侧边栏，登录后进入系统***."
    )
    choice = st.sidebar.selectbox("选择操作", ["登录", "注册"])

    df_users = load_users()

    if choice == "登录":
        username = st.sidebar.text_input("用户名")
        password = st.sidebar.text_input("密码", type='password')
        if st.sidebar.button("登录"):
            if check_user(username, password, df_users):
                st.success("登录成功!")
                # 设置会话状态
                st.session_state['login_status'] = True
                st.session_state['current_user'] = username
                main_page()
            else:
                st.error("用户名或密码错误")

    elif choice == "注册":
        new_username = st.sidebar.text_input("选择一个用户名", key="new_username")
        new_password = st.sidebar.text_input("设置一个密码", type='password', key="new_password")
        confirm_password = st.sidebar.text_input("确认密码", type='password', key="confirm_password")
        if st.sidebar.button("注册"):
            if new_username and new_password:
                if new_password == confirm_password:
                    if not user_exists(new_username, df_users):
                        save_user(new_username, new_password)
                        st.success(f"用户 {new_username} 注册成功!")
                    else:
                        st.error("用户名已经被注册，请选择其他用户名")
                else:
                    st.error("两次输入的密码不匹配")
            else:
                st.error("请输入用户名和密码")

if __name__ == "__main__":
    if 'login_status' not in st.session_state or 'current_user' not in st.session_state:
        st.session_state['login_status'] = False
        st.session_state['current_user'] = ''

    if st.session_state['login_status']:
        main_page()
    else:
        login_page()
    if "index" not in st.session_state:
        st.session_state.index = 0

    if "pre" not in st.session_state:
        st.session_state.pre = 0