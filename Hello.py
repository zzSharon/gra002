
import pandas as pd
import numpy as np
import random
import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)


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
    st.write(test_df.iloc[st.session_state.index])

    genre = st.radio(
        "你的答案是：",
        ["A ::smiling_face_with_3_hearts:", "B ::thermometer:", "C :blush:","D ::sun_with_face:"],
        # captions = ["Laugh out loud.", "Get the popcorn.", "Never stop learning."]
        index=None,)

    if genre == 'A ::smiling_face_with_3_hearts:':
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
    run()
