import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw

st.set_page_config(page_title="太阳能电池板遮挡物检测", page_icon="🔍")
st.title("🔍 太阳能电池板遮挡物检测系统")
st.write("上传图像，一键检测鸟粪、灰尘、落叶、电气损伤、物理损伤、积雪等遮挡物")

# 类别名称（请根据你训练时的顺序调整）
CLASS_NAMES = ['Bird', 'Clean', 'Dust', 'Electrical', 'Physical', 'Snow']

@st.cache_resource
def load_model():
    return YOLO("best.pt")   # 替换为你的纯 m 版权重文件名

model = load_model()

def draw_boxes_pil(image_array, results):
    """使用 PIL 绘制检测框和标签"""
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
