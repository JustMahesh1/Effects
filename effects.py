import cv2
import numpy as np
import streamlit as st
from rembg import remove
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
from scipy.interpolate import UnivariateSpline

# setting app's title, icon & layout
st.set_page_config(page_title="Color Revive - Effects", page_icon="M")

# css style to hide footer, header and main menu details
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {
                visibility: visible;
                position: fixed;
                bottom: 0;
                width: 100%;
                text-align: center;
                font-size: 12px;
                color: white;
                margin: 0;
                padding: 0;
                background: none;
            }
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Add a custom footer with the copyright message
st.markdown(
    """
    <footer>
        <p>&copy; 2025 U Mahesh.</p>
    </footer>
    """,
    unsafe_allow_html=True,
)

def adjust_image_size(image, max_width=600):
    """Resize image maintaining aspect ratio"""
    width, height = image.size
    if width > max_width:
        ratio = max_width / width
        new_height = int(height * ratio)
        return image.resize((max_width, new_height))
    return image

def LookupTable(x, y):
    spline = UnivariateSpline(x, y)
    return spline(range(256))

def cannize_image(input_image):
    rgb_img = np.array(input_image.convert("RGB"))
    img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    img = cv2.GaussianBlur(img, (11, 11), 0)
    edges = cv2.Canny(img, 100, 150)
    return Image.fromarray(edges)

def sepia_effect(input_image):
    rgb_img = np.array(input_image.convert("RGB"))
    img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia = cv2.transform(img, kernel)
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(sepia, cv2.COLOR_BGR2RGB))

def winter_effect(input_image):
    rgb_img = np.array(input_image.convert("RGB"))
    img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    b, g, r = cv2.split(img)
    r = cv2.LUT(r, increaseLookupTable).astype(np.uint8)
    b = cv2.LUT(b, decreaseLookupTable).astype(np.uint8)
    winter = cv2.merge((b, g, r))
    return Image.fromarray(cv2.cvtColor(winter, cv2.COLOR_BGR2RGB))

def summer_effect(input_image):
    rgb_img = np.array(input_image.convert("RGB"))
    img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    b, g, r = cv2.split(img)
    r = cv2.LUT(r, decreaseLookupTable).astype(np.uint8)
    b = cv2.LUT(b, increaseLookupTable).astype(np.uint8)
    summer = cv2.merge((b, g, r))
    return Image.fromarray(cv2.cvtColor(summer, cv2.COLOR_BGR2RGB))

def sketch(input_image):
    image = np.array(input_image.convert("RGB"))
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    invert = cv2.bitwise_not(grey_img)
    blur = cv2.GaussianBlur(invert, (21, 21), 0)
    invertedblur = cv2.bitwise_not(blur)
    sketch_img = cv2.divide(grey_img, invertedblur, scale=256.0)
    return Image.fromarray(sketch_img)

def sharpen_image(input_image):
    image = np.array(input_image.convert("RGB"))
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    kernel = np.array([[-1, -1, -1], 
                       [-1, 9.5, -1], 
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(img, -1, kernel)
    return Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))

def invert_image(input_image):
    image = np.array(input_image.convert("RGB"))
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    inverted = cv2.bitwise_not(img)
    return Image.fromarray(cv2.cvtColor(inverted, cv2.COLOR_BGR2RGB))

def add_custom_background(foreground, background):
    foreground = foreground.convert("RGBA")
    background = background.convert("RGBA").resize(foreground.size)
    composite = Image.new("RGBA", foreground.size)
    composite.paste(background, (0, 0))
    composite.paste(foreground, (0, 0), foreground)
    return composite.convert("RGB")

def generate_meme(image, top_text, bottom_text, font_size=40):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    width, height = img.size
    
    if top_text:
        text_width = draw.textlength(top_text, font)
        draw.text(((width - text_width) / 2, 10), 
                 top_text, font=font, fill="white",
                 stroke_width=3, stroke_fill="black")
    
    if bottom_text:
        text_width = draw.textlength(bottom_text, font)
        draw.text(((width - text_width) / 2, height - font_size - 10),
                 bottom_text, font=font, fill="white",
                 stroke_width=3, stroke_fill="black")
    
    return img

def main():
    st.header("Color Revive - Effects")
    st.text("Blend moods into your images")
    
    image_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    
    enhance_type = st.sidebar.selectbox(
        "Enhance Type",
        [
            "Gray Scale",
            "Pencil Effect",
	    "Background Remover",
            "Custom Background",
            "Sepia Effect",
            "Sharp Effect",
            "Invert Effect",
            "Summer Effect",
            "Winter Effect",
            "Brightness",
            "Blurring",
            "Contrast",
            "Meme Generator",
        ],
    )
    
    if image_file is not None:
        our_image = Image.open(image_file)
        our_image = adjust_image_size(our_image)  # Resize for consistent display
        
        st.text("Original Image")
        st.image(our_image, use_container_width=True)
        
        if enhance_type == "Gray Scale":
            st.text("Filtered Image")
            gray_img = our_image.convert("L")
            st.image(gray_img, use_container_width=True)
            
        elif enhance_type == "Pencil Effect":
            st.text("Filtered Image")
            sketch_img = sketch(our_image)
            st.image(sketch_img, use_container_width=True)
            
        elif enhance_type == "Sepia Effect":
            st.text("Filtered Image")
            sepia_img = sepia_effect(our_image)
            st.image(sepia_img, use_container_width=True)
            
        elif enhance_type == "Sharp Effect":
            st.text("Filtered Image")
            sharp_img = sharpen_image(our_image)
            st.image(sharp_img, use_container_width=True)
            
        elif enhance_type == "Invert Effect":
            st.text("Filtered Image")
            invert_img = invert_image(our_image)
            st.image(invert_img, use_container_width=True)
            
        elif enhance_type == "Summer Effect":
            st.text("Filtered Image")
            summer_img = summer_effect(our_image)
            st.image(summer_img, use_container_width=True)
            
        elif enhance_type == "Winter Effect":
            st.text("Filtered Image")
            winter_img = winter_effect(our_image)
            st.image(winter_img, use_container_width=True)
            
        elif enhance_type == "Background Remover":
            st.text("Filtered Image")
            bg_removed = remove(our_image)
            st.image(bg_removed, use_container_width=True)
            
        elif enhance_type == "Custom Background":
            st.text("Upload a background image")
            bg_file = st.file_uploader("Choose Background Image", type=["jpg", "png", "jpeg"])
            
            if bg_file is not None:
                bg_image = Image.open(bg_file)
                bg_image = adjust_image_size(bg_image)
                st.text("Background Image")
                st.image(bg_image, use_container_width=True)
                
                foreground = remove(our_image)
                result_img = add_custom_background(foreground, bg_image)
                st.text("Result with Custom Background")
                st.image(result_img, use_container_width=True)
                
        elif enhance_type == "Contrast":
            st.text("Filtered Image")
            c_rate = st.sidebar.slider("Contrast", 0.5, 3.5, 1.0)
            enhancer = ImageEnhance.Contrast(our_image)
            img_output = enhancer.enhance(c_rate)
            st.image(img_output, use_container_width=True)
            
        elif enhance_type == "Brightness":
            st.text("Filtered Image")
            b_rate = st.sidebar.slider("Brightness", 0.5, 3.5, 1.0)
            enhancer = ImageEnhance.Brightness(our_image)
            img_output = enhancer.enhance(b_rate)
            st.image(img_output, use_container_width=True)
            
        elif enhance_type == "Blurring":
            st.text("Filtered Image")
            blur_rate = st.sidebar.slider("Blurring", 0.5, 3.5, 1.0)
            img = np.array(our_image.convert("RGB"))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            blur_img = cv2.GaussianBlur(img, (25, 25), blur_rate)
            blur_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB)
            st.image(blur_img, use_container_width=True)
            
        elif enhance_type == "Meme Generator":
            st.text("Create a Meme")
            col1, col2 = st.columns(2)
            
            with col1:
                top_text = st.text_input("Top Text", "")
            with col2:
                bottom_text = st.text_input("Bottom Text", "")
            
            font_size = st.slider("Font Size", 20, 100, 40)
            
            if top_text or bottom_text:
                meme_img = generate_meme(our_image, top_text, bottom_text, font_size)
                st.image(meme_img, use_container_width=True)

if __name__ == "__main__":
    main()
