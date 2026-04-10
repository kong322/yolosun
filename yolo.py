import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import sys
import importlib.util
import builtins

# ================= 注册自定义模块（必须在加载模型之前） =================
def register_custom_modules():
    """查找并注册 MSCAM、SimAM 等自定义类"""
    import os
    module_files = {
        'MSCAM': 'MSCAM.py',
        'SimAM': 'SimAM.py',
        # 如果有其他自定义模块，继续添加
    }
    for class_name, file_name in module_files.items():
        if os.path.exists(file_name):
            spec = importlib.util.spec_from_file_location(class_name, file_name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            builtins.__dict__[class_name] = getattr(module, class_name)
            print(f"✅ 已注册 {class_name}")
        else:
            print(f"⚠️ 未找到 {file_name}，如果模型不需要此模块可忽略")

register_custom_modules()

# ================= 正常加载模型 =================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# 页面配置
st.set_page_config(page_title="YOLO 网页检测", page_icon="🔍")
st.title("🔍 YOLO 目标检测网页系统")
st.write("上传图像，一键检测！")

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
            # 执行推理（如果模型包含自定义模块，此时已注册，不会报错）
            results = model(img_array, conf=0.25)
            res_img = results[0].plot()
            
            with col2:
                st.subheader("检测结果")
                st.image(res_img, use_container_width=True)
            
            st.success(f"检测完成！共发现 {len(results[0].boxes)} 个目标")
