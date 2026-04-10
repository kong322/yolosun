import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# 直接导入自定义类（确保文件存在）
from MSCAM import MSCAM
from SimAM import SimAM

# 可选：将类也注入 builtins（保险起见）
import builtins
builtins.MSCAM = MSCAM
builtins.SimAM = SimAM

# 页面配置
st.set_page_config(page_title="太阳能电池板遮挡物检测", page_icon="🔍")
st.title("🔍 太阳能电池板遮挡物检测系统")
st.write("上传图像，一键检测鸟粪、灰尘、落叶、电气损伤、物理损伤、积雪等遮挡物")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# 后续代码保持不变（使用 cv2 绘图）
uploaded_file = st.file_uploader("选择一张图像", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("原图")
        st.image(image, use_container_width=True)
    if st.button("开始检测"):
        with st.spinner("正在检测..."):
            results = model(img_array, conf=0.25)
            res_img = results[0].plot()  # 使用 cv2 绘图
            with col2:
                st.subheader("检测结果")
                st.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.success(f"检测完成！共发现 {len(results[0].boxes)} 个遮挡物")
