import streamlit as st
import torch
from PIL import Image
import numpy as np
from src.model import Generator  # Import mô hình Generator
from src.utils import convert_text_to_feature

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_dim = 768
noise_dim = 100
embed_out_dim = 64

@st.cache_resource
def load_model():
    generator = Generator(channels=3, embed_dim=embed_dim, 
                            noise_dim=noise_dim, embed_out_dim=embed_out_dim).to(device)
    generator.load_state_dict(torch.load('models/generator_bert.pth'))
    generator.eval() 
    return generator


# Hàm sinh ảnh từ văn bản
def generate_image(text, generator):
    with torch.no_grad():

        embeddings = convert_text_to_feature([str(text)])
        noise = torch.randn(1, noise_dim, 1, 1,device=device)
        pred = generator(noise, embeddings)
    img = pred.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() 
    img = (img * 255).astype(np.uint8)  
    return Image.fromarray(img)


st.set_page_config(page_title="Text-to-Image", layout="wide")

st.title("🖼️ Text-to-Image Synthesis with GAN-CLS + BERT")

# Chia layout với tỷ lệ 1.2 : 1
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("📜 Nhập mô tả hình ảnh")
    text_prompt = st.text_input("", placeholder="Nhập mô tả...", label_visibility="collapsed")

    col1_1, col1_2 = st.columns([0.7, 0.3])
    with col1_1:
        if st.button("🖌️ Generate Image", use_container_width=True):
            if text_prompt.strip() == "":
                st.warning("⚠ Vui lòng nhập mô tả hình ảnh!")
            else:
                generator = load_model()
                output_image = generate_image(text_prompt, generator)

                with col2:
                    st.markdown("✅ **Ảnh đã được tạo thành công!**")
                    st.image(output_image, caption="Generated Image", width=512)
                    

