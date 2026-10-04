# Arabic Manuscript OCR

An academic deep-learning project for converting images of **Arabic handwritten manuscript lines** into digital Arabic text.

The project explores the use of computer vision and deep learning to help digitize old and damaged Arabic manuscripts and make their content easier to search, edit, and preserve.

---

## 🎯 Project Goal

The original project aimed to develop an intelligent system for processing and digitizing old Arabic manuscripts.

The main challenge is that historical manuscripts can contain:

- Degraded or damaged pages
- Uneven backgrounds
- Faded ink
- Noise and stains
- Different handwriting styles
- Closely spaced or overlapping words

Rather than relying only on traditional image-processing techniques, the project investigated deep-learning approaches capable of learning visual and textual representations directly from data.

---

## 🧠 Approach

The project went through several stages before reaching the final image-to-text approach.

### 1. Traditional Image Processing

Several classical techniques were investigated for improving manuscript images and extracting useful structures:

- Static thresholding
- Dynamic thresholding
- Otsu binarization
- Canny edge detection
- Image filters
- Hough transforms
- Mexican Hat filtering
- SIFT
- FAST
- DocEnTR

These approaches were useful for experimentation, but degraded manuscript images presented difficult cases where separating the text from the background was unreliable.

### 2. Word Detection

A contour-based approach was also investigated to extract individual words from manuscript images.

This approach was abandoned because:

- Words could be very close to each other
- Some words could overlap
- Some words could contain or touch other visual regions
- Contour-based segmentation was not reliable across different manuscripts

### 3. Deep Learning

The project then moved toward an image-to-text generation approach.

The main idea was to allow a neural network to learn the relationship between the visual information in a manuscript image and its corresponding Arabic text.

---

## 🏗️ Model Architecture

The developed system combines a CNN-based image feature extractor with a Transformer-based text generation component.

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
