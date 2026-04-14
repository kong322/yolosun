import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(page_title="太阳能电池板遮挡物检测", page_icon="🔍")
st.title("🔍 太阳能电池板遮挡物检测系统")
st.write("上传图像，一键检测鸟粪、灰尘、落叶、电气损伤、物理损伤、积雪等遮挡物")

CLASS_NAMES = ['Bird', 'Clean', 'Dust', 'Electrical', 'Physical', 'Snow']

COLOR_MAP = {
    0: (255, 165, 0),   # Bird - 橙色
    1: (0, 255, 0),     # Clean - 绿色
    2: (128, 128, 128), # Dust - 灰色
    3: (255, 0, 0),     # Electrical - 红色
    4: (128, 0, 128),   # Physical - 紫色
    5: (0, 0, 255)      # Snow - 蓝色
}

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

def draw_boxes(image: Image.Image, results):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except:
        font = ImageFont.load_default()

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        cls_name = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else "Unknown"
        color = COLOR_MAP.get(cls_id, (255, 0, 0))

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        text_x = x1
        text_y = y1 - 18 if y1 > 20 else y2 + 5

        draw.text((text_x, text_y), cls_name, fill=color, font=font)
    return image

uploaded_file = st.file_uploader("选择一张图像", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("原图")
        st.image(image, use_container_width=True)

    if st.button("开始检测"):
        with st.spinner("检测中..."):
            # 关键修复：直接传 PIL 图像，不转 array，避免维度报错
            results = model(image, conf=0.1)
            res_img = draw_boxes(image.copy(), results)

            with col2:
                st.subheader("检测结果")
                st.image(res_img, use_container_width=True)

            st.success(f"检测完成！共 {len(results[0].boxes)} 个目标")
  
