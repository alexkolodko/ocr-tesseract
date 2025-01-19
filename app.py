import streamlit as st
import pytesseract
from PIL import Image
import numpy as np
import cv2
import io
import json

# Set page config
st.set_page_config(layout="wide")

# Add custom CSS to reduce margins/padding
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 0rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        .element-container {
            margin-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

def preprocess_image(image, config):
    """Enhanced image preprocessing with multiple options"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply preprocessing steps based on config
    processed = image.copy()
    
    # Resize if enabled
    if config['resize']['enabled']:
        scale = config['resize']['scale'] / 100
        new_size = (int(processed.shape[1] * scale), int(processed.shape[0] * scale))
        processed = cv2.resize(processed, new_size, interpolation=cv2.INTER_LANCZOS4)
    
    # Denoise if enabled
    if config['denoise']:
        processed = cv2.fastNlMeansDenoising(processed)
    
    # Apply threshold
    if config['threshold']['method'] == 'Binary':
        _, processed = cv2.threshold(processed, config['threshold']['value'], 255, cv2.THRESH_BINARY)
    elif config['threshold']['method'] == 'Adaptive Mean':
        processed = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, config['threshold']['block_size'], 
                                        config['threshold']['C'])
    elif config['threshold']['method'] == 'Adaptive Gaussian':
        processed = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, config['threshold']['block_size'], 
                                        config['threshold']['C'])
    elif config['threshold']['method'] == 'Otsu':
        _, processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply morphological operations if enabled
    if config['morphology']['enabled']:
        kernel_size = config['morphology']['kernel_size']
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        
        if config['morphology']['operation'] == 'Dilate':
            processed = cv2.dilate(processed, kernel, iterations=config['morphology']['iterations'])
        elif config['morphology']['operation'] == 'Erode':
            processed = cv2.erode(processed, kernel, iterations=config['morphology']['iterations'])
        elif config['morphology']['operation'] == 'Open':
            processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
        elif config['morphology']['operation'] == 'Close':
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
    
    # Rotate if enabled
    if config['rotate']['enabled']:
        angle = config['rotate']['angle']
        center = (processed.shape[1] // 2, processed.shape[0] // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        processed = cv2.warpAffine(processed, matrix, (processed.shape[1], processed.shape[0]))
    
    return processed

def main():
    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    
    
    # Create two columns for layout
    left_col, right_col = st.columns([1, 1])
    
    with left_col:
        st.markdown("**Input Image**")
    
    with right_col:
        st.markdown("**Extracted Text**")
    
    # Sidebar options
    st.sidebar.title("Tesseract OCR")
    
    # Language selection with more options
    lang_options = {
        "English": "eng",
        "Ukrainian": "ukr",
        "English + Ukrainian": "eng+ukr",
        "Russian": "rus",
        "Russian + Ukrainian": "rus+ukr",
        "English + Russian + Ukrainian": "eng+rus+ukr"
    }
    selected_lang = st.sidebar.selectbox(
        "Select OCR Language",
        options=list(lang_options.keys())
    )
    
    # OCR Configuration
    st.sidebar.title("OCR Configuration")
    
    # Page segmentation mode with descriptions
    psm_descriptions = {
        0: "Orientation and script detection (OSD) only",
        1: "Automatic page segmentation with OSD",
        2: "Automatic page segmentation, but no OSD, or OCR",
        3: "Fully automatic page segmentation, but no OSD (Default)",
        4: "Assume a single column of text of variable sizes",
        5: "Assume a single uniform block of vertically aligned text",
        6: "Assume a single uniform block of text",
        7: "Treat the image as a single text line",
        8: "Treat the image as a single word",
        9: "Treat the image as a single word in a circle",
        10: "Treat the image as a single character",
        11: "Sparse text. Find as much text as possible in no particular order",
        12: "Sparse text with OSD",
        13: "Raw line. Treat the image as a single text line"
    }
    psm_mode = st.sidebar.selectbox(
        "Page Segmentation Mode",
        options=list(psm_descriptions.keys()),
        format_func=lambda x: f"Mode {x}: {psm_descriptions[x]}",
        index=3
    )
    
    # OCR Engine Mode
    oem_descriptions = {
        0: "Legacy engine only",
        1: "Neural nets LSTM engine only",
        2: "Legacy + LSTM engines",
        3: "Default, based on what is available"
    }
    oem_mode = st.sidebar.selectbox(
        "OCR Engine Mode",
        options=list(oem_descriptions.keys()),
        format_func=lambda x: f"Mode {x}: {oem_descriptions[x]}",
        index=3
    )
    
    # Advanced OCR options
    with st.sidebar.expander("Advanced OCR Options"):
        whitelist_chars = st.text_input("Whitelist Characters", "")
        blacklist_chars = st.text_input("Blacklist Characters", "")
        dpi = st.number_input("DPI", min_value=70, max_value=300, value=70)
    
    # Image preprocessing options
    st.sidebar.title("Image Processing")
    
    # Threshold settings
    threshold_method = st.sidebar.selectbox(
        "Threshold Method",
        ["Binary", "Adaptive Mean", "Adaptive Gaussian", "Otsu"]
    )
    
    threshold_config = {
        'method': threshold_method,
        'value': st.sidebar.slider("Threshold Value", 0, 255, 127) if threshold_method == "Binary" else 0,
        'block_size': st.sidebar.slider("Block Size", 3, 99, 11, step=2) 
            if threshold_method.startswith("Adaptive") else 11,
        'C': st.sidebar.slider("C Value", -50, 50, 2) 
            if threshold_method.startswith("Adaptive") else 2
    }
    
    # Resize settings
    with st.sidebar.expander("Resize Options"):
        resize_config = {
            'enabled': st.checkbox("Enable resize", value=False),
            'scale': st.slider("Scale %", 50, 200, 100)
        }
    
    # Denoising
    denoise = st.sidebar.checkbox("Apply Denoising", value=False)
    
    # Morphological operations
    with st.sidebar.expander("Morphological Operations"):
        morph_config = {
            'enabled': st.checkbox("Enable morphology", value=False),
            'operation': st.selectbox("Operation", ["Dilate", "Erode", "Open", "Close"]),
            'kernel_size': st.slider("Kernel Size", 1, 9, 3, step=2),
            'iterations': st.slider("Iterations", 1, 5, 1)
        }
    
    # Rotation
    with st.sidebar.expander("Rotation Options"):
        rotate_config = {
            'enabled': st.checkbox("Enable rotation", value=False),
            'angle': st.slider("Angle", -180, 180, 0)
        }
    
    # Output format options
    st.sidebar.title("Output Options")
    output_format = st.sidebar.selectbox(
        "Output Format",
        ["Plain Text", "JSON with Confidence", "JSON with Boxes", "JSON with Words"]
    )
    

    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Show original image
        with left_col:
            st.image(image, caption="Original Image", use_column_width=True)
        
        # Preprocess image with all options
        preprocessing_config = {
            'threshold': threshold_config,
            'resize': resize_config,
            'denoise': denoise,
            'morphology': morph_config,
            'rotate': rotate_config
        }
        
        processed_img = preprocess_image(img_array, preprocessing_config)
        with left_col:
            st.image(processed_img, caption="Preprocessed Image", use_column_width=True)
        
        # Perform OCR
        try:
            # Build tesseract config
            custom_config = f'--psm {psm_mode} --oem {oem_mode}'
            if whitelist_chars:
                custom_config += f' -c tessedit_char_whitelist={whitelist_chars}'
            if blacklist_chars:
                custom_config += f' -c tessedit_char_blacklist={blacklist_chars}'
            custom_config += f' --dpi {dpi}'
            
            # Get OCR result based on output format
            if output_format == "Plain Text":
                result = pytesseract.image_to_string(
                    processed_img,
                    lang=lang_options[selected_lang],
                    config=custom_config
                )
                display_text = result
            elif output_format == "JSON with Confidence":
                data = pytesseract.image_to_data(
                    processed_img,
                    lang=lang_options[selected_lang],
                    config=custom_config,
                    output_type=pytesseract.Output.DICT
                )
                result = {
                    'text': " ".join([word for word in data['text'] if word.strip()]),
                    'confidence': data['conf']
                }
                display_text = json.dumps(result, indent=2)
            elif output_format == "JSON with Boxes":
                boxes = pytesseract.image_to_boxes(
                    processed_img,
                    lang=lang_options[selected_lang],
                    config=custom_config
                )
                display_text = boxes
            else:  # JSON with Words
                data = pytesseract.image_to_data(
                    processed_img,
                    lang=lang_options[selected_lang],
                    config=custom_config,
                    output_type=pytesseract.Output.DICT
                )
                words = []
                for i in range(len(data['text'])):
                    if data['text'][i].strip():
                        words.append({
                            'text': data['text'][i],
                            'confidence': data['conf'][i],
                            'box': {
                                'x': data['left'][i],
                                'y': data['top'][i],
                                'width': data['width'][i],
                                'height': data['height'][i]
                            }
                        })
                display_text = json.dumps({'words': words}, indent=2)
            
            # Display result
            with right_col:
                st.text_area("Extracted Text", display_text, height=400)
                
                # Add download button for text
                if display_text:
                    file_extension = '.json' if output_format != "Plain Text" else '.txt'
                    st.download_button(
                        label="Download result",
                        data=display_text,
                        file_name=f"extracted_text{file_extension}",
                        mime="application/json" if output_format != "Plain Text" else "text/plain"
                    )
        
        except Exception as e:
            st.error(f"Error during OCR: {str(e)}")
            st.info("Make sure you have Tesseract installed and the language packs are available.")

if __name__ == "__main__":
    main()
