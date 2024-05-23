import streamlit as st

# 假设用户数据存储在这里（在实际应用中，请使用数据库并加密密码）
users = {
    "user1": "password1",
    "user2": "password2"
}

session_state = st.session_state
if 'login_status' not in session_state:
    session_state.login_status = False
if 'current_user' not in session_state:
    session_state.current_user = ''

def login_user(username, password):
    """校验用户登录信息"""
    return users.get(username) == password

def main_page():
    st.write(f"欢迎 {session_state.current_user}, 你已成功登录！")
    # 这里添加登录成功后显示的页面内容
    st.write("这是登录后用户看到的页面内容。")

def login_page():
    st.title("简易登录注册系统")

    # 创建一个选择器让用户选择登录或注册
    choice = st.sidebar.selectbox("登录或注册", ["登录", "注册"])

    if choice == "登录":
        username = st.sidebar.text_input("用户名")
        password = st.sidebar.text_input("密码", type='password')
        if st.sidebar.button("登录"):
            if login_user(username, password):
                st.success("登录成功!")
                session_state.login_status = True
                session_state.current_user = username
            else:
                st.error("用户名或密码错误")

    elif choice == "注册":
        new_username = st.sidebar.text_input("选择一个用户名")
        new_password = st.sidebar.text_input("设置一个密码", type='password')
        confirm_password = st.sidebar.text_input("确认密码", type='password')
        if st.sidebar.button("注册"):
            if new_password == confirm_password:
                # 在实际应用中，应该在这里将新用户数据安全存储起来
                st.success(f"用户 {new_username} 注册成功!")
                users[new_username] = new_password
            else:
                st.error("两次输入的密码不匹配")

if __name__ == "__main__":
    if session_state.login_status:
        main_page()
    else:
        login_page()
