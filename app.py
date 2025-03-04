import streamlit as st
import torch
from PIL import Image
import numpy as np
from src.model import Generator
from src.sr_model import SR_Unet  # Import mô hình Generator
from src.utils import convert_text_to_feature
import torch.hub
import gdown
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_dim = 768
noise_dim = 100
embed_out_dim = 64

@st.cache_resource
def load_model():
    generator = Generator(channels=3, embed_dim=embed_dim, 
                          noise_dim=noise_dim, embed_out_dim=embed_out_dim).to(device)
    sr_unet = SR_Unet().to(device)

    # Tải mô hình từ Google Drive nếu chưa có
    unet_path = "models/sr_unet.pth"
    unet_url = "https://drive.google.com/uc?export=download&id=1hEGvRbJq4G_LVGSd1ZozA8UUMJzG1MBp"
    
    if not os.path.exists(unet_path):
        gdown.download(unet_url, unet_path, quiet=False)

    if device == "cuda":
        generator.load_state_dict(torch.load("models/generator_bert.pth"))
        sr_unet.load_state_dict(torch.load(unet_path))
    else:
        generator.load_state_dict(torch.load("models/generator_bert.pth", map_location="cpu"))
        sr_unet.load_state_dict(torch.load(unet_path, map_location="cpu"))

    generator.eval()
    sr_unet.eval()
    return generator, sr_unet


def generate_image(text, generator, sr_unet):
    with torch.no_grad():
        embeddings = convert_text_to_feature([str(text)])
        noise = torch.randn(1, noise_dim, 1, 1, device=device)

        # Ảnh từ GAN
        gen_img = generator(noise, embeddings)

        # Ảnh sau khi qua Super-Resolution UNet
        pred = sr_unet(gen_img)

    # Chuyển ảnh từ tensor sang định dạng có thể hiển thị
    gen_img_np = gen_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    gen_img_np = (gen_img_np * 255).astype(np.uint8)

    pred_np = pred.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    pred_np = (pred_np * 255).astype(np.uint8)

    return Image.fromarray(gen_img_np), Image.fromarray(pred_np)





st.set_page_config(page_title="Text-to-Image", layout="wide")

st.title("🖼️ Text-to-Image Synthesis with GAN-CLS + BERT")

# Divide into 3 equal parts
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📜 Enter image description")
    text_prompt = st.text_input("", placeholder="Enter description...", label_visibility="collapsed")

    if st.button("🖌️ Generate Image", use_container_width=True):
        if text_prompt.strip() == "":
            st.warning("⚠ Please enter an image description!")
        else:
            generator, sr_unet = load_model()
            gen_image, sr_image = generate_image(text_prompt, generator, sr_unet)

            with col2:
                st.image(gen_image, caption="Generated Image", width=512)

            with col3:
                st.image(sr_image, caption="Generated Image + Super Resolution", width=512)

                    

