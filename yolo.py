import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw
import os
import importlib.util
import builtins

# ========== 注册自定义模块（如果模型包含 MSCAM、SimAM 等） ==========
def register_custom_modules():
    module_files = {
        'MSCAM': 'MSCAM.py',
        'SimAM': 'SimAM.py',
    }
    for class_name, file_name in module_files.items():
        if os.path.exists(file_name):
            spec = importlib.util.spec_from_file_location(class_name, file_name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            builtins.__dict__[class_name] = getattr(module, class_name)

register_custom_modules()

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.set_page_config(page_title="太阳能电池板遮挡物检测", page_icon="🔍")
st.title("🔍 太阳能电池板遮挡物检测系统")
st.write("上传图像，一键检测鸟粪、灰尘、落叶、电气损伤、物理损伤、积雪等遮挡物")

def draw_boxes_pil(image_array, results):
    img = Image.fromarray(image_array).convert('RGB')
    draw = ImageDraw.Draw(img)
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        label = f"{model.names[cls]}: {conf:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1-10), label, fill="red")
    return img

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
            result_img = draw_boxes_pil(img_array, results)
            with col2:
                st.subheader("检测结果")
                st.image(result_img, use_container_width=True)
            st.success(f"检测完成！共发现 {len(results[0].boxes)} 个遮挡物")
