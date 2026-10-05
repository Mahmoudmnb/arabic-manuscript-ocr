# Arabic Manuscript OCR

An academic deep-learning project for converting images of **Arabic handwritten manuscript lines** into digital Arabic text.

The project explores computer vision and deep learning techniques for helping digitize old and damaged Arabic manuscripts and making their content easier to search, edit, and preserve.

---

## 🎯 Project Goal

The project aimed to develop an intelligent system for processing and digitizing old Arabic manuscripts.

Historical manuscripts can contain challenging visual conditions such as:

- Degraded or damaged pages
- Uneven backgrounds
- Faded ink
- Noise and stains
- Different handwriting styles
- Closely spaced or overlapping words

Several image-processing approaches were investigated before moving toward a deep-learning image-to-text system.

---

## 🔬 Development Approach

The project went through several stages during development.

### 1. Traditional Image Processing

Several classical image-processing techniques were investigated for improving manuscript images and extracting useful information:

- Static thresholding
- Dynamic thresholding
- Otsu binarization
- Canny edge detection
- Image filters
- Hough transform
- Mexican Hat filter
- SIFT
- FAST
- DocEnTR

These approaches were useful for experimentation, but old manuscript images introduced difficult backgrounds, stains, faded ink, and other variations that made reliable text extraction challenging.

### 2. Contour-Based Word Extraction

A contour-based approach was also investigated to extract individual words from manuscript images.

This approach was not reliable because:

- Words could be very close to each other
- Some words could overlap
- Different regions could touch each other
- Contours were not consistently suitable for separating manuscript words

Because of these limitations, the project moved toward a learned image-to-text approach.

### 3. Deep Learning Image-to-Text

The project then explored an image-to-text generation system capable of learning the relationship between manuscript images and their corresponding Arabic text.

The main architecture combines:

- A CNN-based visual feature extractor
- A Transformer-based decoder
- Self-attention
- Cross-attention between text generation and visual features

---

## 🏗️ Model Architecture

The developed system follows an image-to-text architecture:

```mermaid
flowchart LR
    A[Arabic Manuscript Line Image] --> B[Image Preprocessing]
    B --> C[CNN Feature Extractor]
    C --> D[Visual Feature Representation]
    D --> E[Transformer Decoder]

    E --> F[Self-Attention]
    E --> G[Cross-Attention]

    F --> H[Arabic Text Tokens]
    G --> H

    H --> I[Generated Arabic Text]
