import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw
import torch
import builtins
import os

# ========== 注册自定义模块（必须放在加载模型之前） ==========
def register_custom_modules():
    """动态导入 MSCAM 和 SimAM 类，并注入 builtins"""
    try:
        from MSCAM import MSCAM
        builtins.MSCAM = MSCAM
        print("✅ MSCAM 注册成功")
    except ImportError:
        print("⚠️ 未找到 MSCAM.py，如果模型不需要此模块可忽略")
    try:
        from SimAM import SimAM
        builtins.SimAM = SimAM
        print("✅ SimAM 注册成功")
    except ImportError:
        print("⚠️ 未找到 SimAM.py，如果模型不需要此模块可忽略")

register_custom_modules()

# ========== 加载模型 ==========
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# 类别名称（根据训练时的顺序：Bird, Clean, Dust, Electrical, Physical, Snow）
CLASS_NAMES = model.names if hasattr(model, 'names') else {
    0: 'Bird', 1: 'Clean', 2: 'Dust', 3: 'Electrical', 4: 'Physical', 5: 'Snow'
}

# ========== 绘图函数（纯 PIL，无 cv2） ==========
def draw_boxes_pil(image_array, results):
    img = Image.fromarray(image_array).convert('RGB')
    draw = ImageDraw.Draw(img)
    boxes = results[0].boxes
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{CLASS_NAMES[cls]}: {conf:.2f}"
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1-10), label, fill="red")
    return img

# ========== Streamlit UI ==========
st.set_page_config(page_title="太阳能电池板遮挡物检测", page_icon="🔍")
st.title("🔍 太阳能电池板遮挡物检测系统")
st.write("上传图像，一键检测鸟粪、灰尘、落叶、电气损伤、物理损伤、积雪等遮挡物")

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
            # 推理
            results = model(img_array, conf=0.25)
            # 绘制结果
            result_img = draw_boxes_pil(img_array, results)
            with col2:
                st.subheader("检测结果")
                st.image(result_img, use_container_width=True)
            st.success(f"检测完成！共发现 {len(results[0].boxes)} 个遮挡物")
