# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.hello.utils import show_code


def NCR_Detail() -> None:

    # Interactive Streamlit elements, like these sliders, return their value.
    # This gives you an extremely simple interaction model.
    iterations = st.sidebar.slider("Level of detail", 2, 20, 10, 1)
    separation = st.sidebar.slider("Separation", 0.7, 2.0, 0.7885)

    # Non-interactive elements return a placeholder to their location
    # in the app. Here we're storing progress_bar to update it later.
    progress_bar = st.sidebar.progress(0)

    # These two elements will be filled in later, so we create a placeholder
    # for them using st.empty()
    frame_text = st.sidebar.empty()
    image = st.empty()

    m, n, s = 960, 640, 400
    x = np.linspace(-m / s, m / s, num=m).reshape((1, m))
    y = np.linspace(-n / s, n / s, num=n).reshape((n, 1))

    for frame_num, a in enumerate(np.linspace(0.0, 4 * np.pi, 100)):
        # Here were setting value for these two elements.
        progress_bar.progress(frame_num)
        frame_text.text("Frame %i/100" % (frame_num + 1))

        # Performing some fractal wizardry.
        c = separation * np.exp(1j * a)
        Z = np.tile(x, (n, 1)) + 1j * np.tile(y, (1, m))
        C = np.full((n, m), c)
        M: Any = np.full((n, m), True, dtype=bool)
        N = np.zeros((n, m))

        for i in range(iterations):
            Z[M] = Z[M] * Z[M] + C[M]
            M[np.abs(Z) > 2] = False
            N[M] = i

        # Update the image placeholder by calling the image() function on it.
        image.image(1.0 - (N / N.max()), use_column_width=True)

    # We clear elements by calling empty on them.
    progress_bar.empty()
    frame_text.empty()

    # Streamlit widgets automatically run the script from top to bottom. Since
    # this button is not connected to any other logic, it just causes a plain
    # rerun.
    st.button("Re-run")


st.set_page_config(page_title="NCR_Detail😃", page_icon="📹")
st.markdown("# NCR_Detail😃")
st.sidebar.header("NCR_Detail")
st.sidebar.header("这里展示了NCR数据集的难度分布与文体分布，同时还可以为系统添加数据。题目具有相当的难度，且文体包含文言文、诗歌等多种文体。")
st.write(
    """这里展示了NCR数据集的更多信息:."""
)
st.markdown(
" ***👇 下面是数据集的难度统计***."
)

test_df = pd.read_csv('/workspaces/gra002/test.csv')
diff_counts = test_df['Diff'].value_counts()

# 创建一个包含diff_counts的DataFrame，使其成为一行多列
diff_counts_df = pd.DataFrame([diff_counts])

# 由于diff_counts已经作为Series返回了DIFF类别作为索引，这里直接转换成DataFrame时会保持索引
# 转置DataFrame，使其成为一行多列
diff_counts_df = diff_counts_df.T.reset_index()  # 重置索引来将原来的索引（DIFF类别）变成一列
diff_counts_df.columns = ['DIFF', 'Counts']      # 重命名列名

# 如果你想要一个一行七列的DataFrame（不包括DIFF类别作为数据列），那么可以这样操作
one_row_df = pd.DataFrame([diff_counts.values], columns=diff_counts.index)
chart_data = pd.DataFrame(one_row_df)
# st.bar_chart(chart_data)
# plt.bar(x=test_df.Diff,height=diff_counts)
# 提供的DataFrame数据
data = {'Diff': [6, 8, 7, 1, 3, 4, 2], '0': [1940, 264, 131, 48, 29, 22, 10]}
one_row_df = pd.DataFrame(data)

# 设置Diff为x轴的标签
x_labels = one_row_df['Diff'].tolist()

# 设置数值为y轴的值
y_values = one_row_df['0'].tolist()

import matplotlib.pyplot as plt
# 创建条形图

fig=plt.figure()
plt.bar(x_labels, y_values, color='orangered')

# 设置x轴的标题
plt.xlabel('DIFF')

# 设置y轴的标题
plt.ylabel('Values')

# 添加图表的标题
plt.title('Bar Chart of DIFF Values')

# 展示图表
st.pyplot(fig)


st.markdown(
" ***👇 下面是数据集的类别统计***."
)

# 提取'type'列作为x轴的标签
type = test_df['Type'].value_counts()
# 提取与'type'相关的值作为y轴的值
# 这假设DataFrame有一个与每个'type'相关的数值列，此处命名为'value'
# 假设这是你的DataFrame
data = {'Type': [0, 11, 22, 33], 'Count': [1842, 543, 49, 10]}
test_df = pd.DataFrame(data)

fig=plt.figure()
# 设置Type为x轴的标签
x_labels = test_df['Type'].tolist()

# 设置Count为y轴的值
y_values = test_df['Count'].tolist()

# 创建条形图
plt.bar(x_labels, y_values,color='orange')

# 设置x轴的标题
plt.xlabel('Type')

# 设置y轴的标题
plt.ylabel('Count')

# 添加图表的标题
plt.title('Bar Chart of Counts by Type')
st.pyplot(fig)

test_df = pd.read_csv('/workspaces/gra002/test.csv')
# a=test_df['Questions']['Answer'].value_counts()
# print(test_df.iloc[3])

# show_code(animation_demo)

def set_bg_hack_url():
    '''
    A function to unpack an image from url and set as bg.
    Returns
    -------
    The background.
    '''
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://i.postimg.cc/kXKdK0K9/tomer-texler-MIGq1g0ws7k-unsplash.jpg");
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
st.write(test_df.iloc[0])
import streamlit as st
import csv

# 指定CSV文件路径
csv_file_path = '/workspaces/gra002/test.csv'

# 创建Streamlit表单
with st.form("my_form"):
    st.write("在这里添加数据:")
    
    # 创建输入框让用户输入每个字段的值
    question = st.text_input("问题")
    choices = st.text_input("选项")
    q_id = st.text_input("题目id")
    content = st.text_input("文章")
    type_ = st.text_input("文体")
    diff = st.text_input("难度")
    
    # 每当用户完成表单并点击"Submit"按钮时，就会提交表单
    submitted = st.form_submit_button("提交")
    if submitted:
        # 数据行
        new_row = {
            'Question': question,
            'Choices': choices,
            'Q_id': q_id,
            'Content': content,
            'Type': type_,
            'Diff': diff
        }
        
        # 打开CSV文件准备写入
        with open(csv_file_path, mode='a', newline='', encoding='utf-8') as file:
            # 创建一个DictWriter对象
            writer = csv.DictWriter(file, fieldnames=['Question', 'Choices', 'Q_id', 'Content', 'Type', 'Diff'])
            
            # 写入新的数据行
            writer.writerow(new_row)
        
        # 显示操作成功的消息
        st.success("数据添加成功!")
import csv

# 指定CSV文件路径
csv_file_path = 'test.csv'

# 打开CSV文件准备读取
with open(csv_file_path, mode='r', newline='') as file:
    # 创建一个csv.reader对象
    reader = csv.reader(file)
    
    # 初始化last_row变量
    last_row = None
    
    # 逐行读取文件内容直到最后一行
    for row in reader:
        last_row = row

# # # 打印最后一行数据
# st.write(last_row)
# test_df = pd.read_csv('/workspaces/gra002/test.csv')
# st.write(len(test_df))
# unique_types = test_df['Diff'].unique()

# # 使用 st.selectbox 创建一个下拉选择框
# selected_value = st.selectbox("请从以下数值中选择一个：", unique_types)
# print(unique_types)
# test_df = test_df[test_df['Diff'] == selected_value]
# st.write(test_df)

set_bg_hack_url()

