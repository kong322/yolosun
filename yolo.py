import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# 页面配置
st.set_page_config(page_title="YOLO 网页检测", page_icon="🔍")

# 加载模型（缓存，避免重复加载）
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# 网页标题
st.title("🔍 YOLO 目标检测网页系统")
st.write("上传图像，一键检测！")

# 1. 上传图像
uploaded_file = st.file_uploader("选择一张图像", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 读取图像
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # 显示原图
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("原图")
        st.image(image, use_container_width=True)
    
    # 2. 检测按钮
    if st.button("开始检测"):
        with st.spinner("正在检测..."):
            # 执行推理
            results = model(img_array, conf=0.25)
            res_img = results[0].plot()
            
            # 显示结果
            with col2:
                st.subheader("检测结果")
                st.image(res_img, use_container_width=True)
            
            # 显示检测数量
            st.success(f"检测完成！共发现 {len(results[0].boxes)} 个目标")