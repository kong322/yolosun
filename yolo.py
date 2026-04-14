import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(page_title="太阳能电池板遮挡物检测", page_icon="🔍")
st.title("🔍 太阳能电池板遮挡物检测系统")
st.write("上传图像，一键检测鸟粪、灰尘、落叶、电气损伤、物理损伤、积雪等遮挡物")

# 【重要】请严格按照你训练时的类别顺序排列！
# 如果不确定，先运行 print(model.names) 查看
CLASS_NAMES = ['Bird', 'Clean', 'Dust', 'Electrical', 'Physical', 'Snow']

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

def draw_boxes_pil(image_array, results):
    """使用 PIL 绘制检测框和带背景的标签"""
    img = Image.fromarray(image_array).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    # 尝试加载系统默认字体，如果失败则使用默认位图字体
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    boxes = results[0].boxes
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            # 安全检查：防止类别索引越界
            if cls < 0 or cls >= len(CLASS_NAMES):
                class_name = "Unknown"
            else:
                class_name = CLASS_NAMES[cls]
            
            label = f"{class_name} {conf:.2f}"
            
            # 1. 绘制红色边界框
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            
            # 2. 计算标签背景框尺寸
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # 3. 调整标签位置：如果y1太靠上，就把标签放在框下面
            if y1 - text_height - 10 > 0:
                text_y = y1 - text_height - 5
            else:
                text_y = y2 + 5  # 放在框下面
            
            text_x = x1
            
            # 4. 绘制白色背景框（让文字更清晰）
            draw.rectangle(
                [text_x - 2, text_y - 2, text_x + text_width + 2, text_y + text_height + 2],
                fill="white"
            )
            
            # 5. 绘制红色文字
            draw.text((text_x, text_y), label, fill="red", font=font)
    
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
            results = model(img_array, conf=0.1)
            result_img = draw_boxes_pil(img_array, results)
            with col2:
                st.subheader("检测结果")
                st.image(result_img, use_container_width=True)
            st.success(f"检测完成！共发现 {len(results[0].boxes)} 个遮挡物")
