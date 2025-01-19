# Tesseract OCR App

A Streamlit web application for performing OCR (Optical Character Recognition) on images and PDFs using Tesseract, with support for English and Ukrainian languages.

## Prerequisites

1. Install Tesseract OCR on your system:
   - **macOS**: `brew install tesseract tesseract-lang poppler`
   - **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr tesseract-ocr-eng tesseract-ocr-ukr poppler-utils`
   - **Windows**: 
     - Download Tesseract installer from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
     - Download and install poppler from [here](http://blog.alivate.com.au/poppler-windows/)

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Running the App

```bash
streamlit run app.py
```

## Features

- Support for both images (.png, .jpg, .jpeg) and PDFs
- PDF processing with page selection and DPI control
- Support for English and Ukrainian text recognition
- Multiple output formats:
  - Plain text
  - JSON with confidence scores
  - JSON with bounding boxes
  - JSON with detailed word information
- Advanced image preprocessing options:
  - Multiple thresholding methods
  - Image resizing
  - Denoising
  - Morphological operations
  - Rotation control
- OCR Configuration:
  - Page Segmentation Mode (PSM)
  - OCR Engine Mode (OEM)
  - Character whitelist/blacklist
  - DPI settings

## Usage

1. Launch the app using the command above
2. Upload an image or PDF file
3. For PDFs:
   - Select pages to process
   - Adjust PDF to image conversion DPI
4. Select the language(s) for OCR
5. Adjust preprocessing settings if needed
6. Choose output format
7. View the extracted text and download results 