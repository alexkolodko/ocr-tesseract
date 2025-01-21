import streamlit as st
import pytesseract
from PIL import Image
import numpy as np
import cv2
import io
import json
import pdf2image
import tempfile
import os
import sys

# Set page config
st.set_page_config(layout="wide")

# Check Tesseract installation
try:
    tesseract_version = pytesseract.get_tesseract_version()
    st.sidebar.success(f"✓ Tesseract version: {tesseract_version}")
except Exception as e:
    st.sidebar.error(f"❌ Tesseract not found: {str(e)}")
    st.sidebar.info("Please install Tesseract and make sure it's in your PATH")

# Check poppler installation for PDF support
try:
    if sys.platform.startswith('win'):
        if not os.environ.get('POPPLER_PATH'):
            st.sidebar.warning("⚠️ POPPLER_PATH not set. PDF support may not work.")
    else:
        # On Unix systems, try to find poppler-utils
        from shutil import which
        if which('pdftoppm') is None:
            st.sidebar.warning("⚠️ poppler-utils not found. PDF support may not work.")
        else:
            st.sidebar.success("✓ poppler-utils found")
except Exception as e:
    st.sidebar.warning(f"⚠️ Could not verify poppler installation: {str(e)}")

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

def process_file(file, preprocessing_config, ocr_config, output_format):
    """Process a single image or page and return OCR results"""
    try:
        # Convert to numpy array
        image = Image.open(file) if isinstance(file, (str, bytes, io.BytesIO)) else file
        img_array = np.array(image)
        
        # Preprocess image
        processed_img = preprocess_image(img_array, preprocessing_config)
        
        # Perform OCR with config
        if output_format == "Plain Text":
            result = pytesseract.image_to_string(
                processed_img,
                lang=ocr_config['lang'],
                config=ocr_config['config']
            )
            return result, processed_img
        elif output_format == "JSON with Confidence":
            data = pytesseract.image_to_data(
                processed_img,
                lang=ocr_config['lang'],
                config=ocr_config['config'],
                output_type=pytesseract.Output.DICT
            )
            result = {
                'text': " ".join([word for word in data['text'] if word.strip()]),
                'confidence': data['conf']
            }
            return json.dumps(result, indent=2), processed_img
        elif output_format == "JSON with Boxes":
            boxes = pytesseract.image_to_boxes(
                processed_img,
                lang=ocr_config['lang'],
                config=ocr_config['config']
            )
            return boxes, processed_img
        else:  # JSON with Words
            data = pytesseract.image_to_data(
                processed_img,
                lang=ocr_config['lang'],
                config=ocr_config['config'],
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
            return json.dumps({'words': words}, indent=2), processed_img
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None

def main():
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a file", type=["png", "jpg", "jpeg", "pdf"])
    
    # Create two columns for layout
    left_col, right_col = st.columns([1, 1])
    
    with left_col:
        st.markdown("**Document Pages**")
    
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
        "Russian + English": "rus+eng",
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
    
    # PDF specific options
    pdf_options = {}
    if uploaded_file is not None and uploaded_file.name.lower().endswith('.pdf'):
        st.sidebar.title("PDF Options")
        
        try:
            # Save PDF to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                pdf_path = tmp_file.name
            
            # Get number of pages
            try:
                pdf_pages = pdf2image.pdfinfo_from_path(pdf_path)
                n_pages = pdf_pages["Pages"]
                st.sidebar.info(f"PDF pages detected: {n_pages}")
            except Exception as e:
                st.error(f"Error reading PDF info: {str(e)}")
                st.info("Please make sure poppler is properly installed")
                return
            
            # DPI for PDF conversion
            pdf_options['dpi'] = st.sidebar.slider("PDF to Image DPI", 100, 600, 200, 50)
            
            # Process all pages
            if uploaded_file is not None:
                # Prepare OCR configuration
                ocr_config = {
                    'lang': lang_options[selected_lang],
                    'config': f'--psm {psm_mode} --oem {oem_mode}'
                }
                if whitelist_chars:
                    ocr_config['config'] += f' -c tessedit_char_whitelist={whitelist_chars}'
                if blacklist_chars:
                    ocr_config['config'] += f' -c tessedit_char_blacklist={blacklist_chars}'
                ocr_config['config'] += f' --dpi {dpi}'
                
                # Prepare preprocessing configuration
                preprocessing_config = {
                    'threshold': threshold_config,
                    'resize': resize_config,
                    'denoise': denoise,
                    'morphology': morph_config,
                    'rotate': rotate_config
                }
                
                # Convert PDF pages to images and process them
                try:
                    with st.spinner(f"Processing all {n_pages} pages..."):
                        st.info("Converting PDF to images...")
                        images = pdf2image.convert_from_path(
                            pdf_path,
                            dpi=pdf_options['dpi']
                        )
                        st.success(f"Successfully converted {len(images)} pages")
                        
                        all_results = []
                        all_processed_images = []
                        
                        progress_bar = st.progress(0)
                        for idx, image in enumerate(images, start=1):
                            st.info(f"Processing page {idx}/{len(images)}")
                            result, processed_img = process_file(
                                image,
                                preprocessing_config,
                                ocr_config,
                                output_format
                            )
                            if result:
                                all_results.append((idx, result, processed_img))
                                st.success(f"✓ Page {idx} processed")
                            else:
                                st.error(f"❌ Failed to process page {idx}")
                            progress_bar.progress(idx / len(images))
                        progress_bar.empty()
                
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                    st.info("Try adjusting the DPI value or check if poppler is properly installed")
                    return
                
                # Display results
                if all_results:
                    # Create tabs for page images in left column
                    with left_col:
                        tabs = st.tabs([f"Page {idx}" for idx, _, _ in all_results])
                        for tab, (idx, _, processed_img) in zip(tabs, all_results):
                            with tab:
                                st.image(processed_img, use_column_width=True)
                    
                    # Show all text in right column
                    with right_col:
                        # Join all text without page dividers
                        if output_format == "Plain Text":
                            combined_text = "\n\n".join(result for _, result, _ in all_results)
                        else:
                            # For JSON formats, combine results into a single JSON structure
                            if output_format == "JSON with Confidence":
                                combined_results = {
                                    "pages": [
                                        {
                                            "page": idx,
                                            "content": json.loads(result)
                                        } for idx, result, _ in all_results
                                    ]
                                }
                                combined_text = json.dumps(combined_results, indent=2)
                            elif output_format == "JSON with Words":
                                combined_results = {
                                    "pages": [
                                        {
                                            "page": idx,
                                            "content": json.loads(result)
                                        } for idx, result, _ in all_results
                                    ]
                                }
                                combined_text = json.dumps(combined_results, indent=2)
                            else:  # JSON with Boxes
                                combined_text = "\n\n".join(f"Page {idx}:\n{result}" for idx, result, _ in all_results)
                        
                        st.text_area("Extracted Text", combined_text, height=800)
                    
                    # Add download button
                    st.sidebar.markdown("---")
                    file_extension = '.json' if output_format != "Plain Text" else '.txt'
                    st.sidebar.download_button(
                        label="Download All Pages",
                        data=combined_text,
                        file_name=f"extracted_text{file_extension}",
                        mime="application/json" if output_format != "Plain Text" else "text/plain"
                    )
                else:
                    st.warning("No results were generated. Please check the error messages above.")
            
            # Cleanup temporary file
            try:
                os.unlink(pdf_path)
            except Exception as e:
                st.warning(f"Could not remove temporary file: {str(e)}")
        
        except Exception as e:
            st.error(f"Error handling PDF file: {str(e)}")
            st.info("Please make sure the PDF file is valid and not corrupted")
    
    # Process image files
    elif uploaded_file is not None:
        # Prepare OCR configuration
        ocr_config = {
            'lang': lang_options[selected_lang],
            'config': f'--psm {psm_mode} --oem {oem_mode}'
        }
        if whitelist_chars:
            ocr_config['config'] += f' -c tessedit_char_whitelist={whitelist_chars}'
        if blacklist_chars:
            ocr_config['config'] += f' -c tessedit_char_blacklist={blacklist_chars}'
        ocr_config['config'] += f' --dpi {dpi}'
        
        # Prepare preprocessing configuration
        preprocessing_config = {
            'threshold': threshold_config,
            'resize': resize_config,
            'denoise': denoise,
            'morphology': morph_config,
            'rotate': rotate_config
        }
        
        # Process single image
        result, processed_img = process_file(
            uploaded_file,
            preprocessing_config,
            ocr_config,
            output_format
        )
        
        if result and processed_img is not None:
            # Show processed image in left column
            with left_col:
                st.image(processed_img, caption="Processed Image", use_column_width=True)
            
            # Show text in right column
            with right_col:
                st.text_area("Extracted Text", result, height=400)
                
                # Add download button for text
                file_extension = '.json' if output_format != "Plain Text" else '.txt'
                st.download_button(
                    label="Download result",
                    data=result,
                    file_name=f"extracted_text{file_extension}",
                    mime="application/json" if output_format != "Plain Text" else "text/plain"
                )

if __name__ == "__main__":
    main()
