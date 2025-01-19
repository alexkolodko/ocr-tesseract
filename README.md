# Tesseract OCR App

A Streamlit web application for performing OCR (Optical Character Recognition) on images using Tesseract, with support for English and Ukrainian languages.

## Prerequisites

1. Install Tesseract OCR on your system:
   - **macOS**: `brew install tesseract tesseract-lang`
   - **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr tesseract-ocr-eng tesseract-ocr-ukr`
   - **Windows**: Download installer from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Running the App

```bash
streamlit run app.py
```

## Features

- Support for English and Ukrainian text recognition
- Image preprocessing options
- Adjustable Page Segmentation Mode (PSM)
- Download extracted text as file
- Side-by-side view of original/processed image and extracted text

## Usage

1. Launch the app using the command above
2. Upload an image containing text (.png, .jpg, or .jpeg)
3. Select the language(s) for OCR
4. Adjust preprocessing settings if needed
5. View the extracted text and download if desired 