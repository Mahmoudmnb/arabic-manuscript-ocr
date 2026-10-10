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

### From Manuscript Image to Digital Text

<p align="center">
  <img src="docs/images/01-input-to-text.png" alt="Arabic manuscript image to digital text example" width="95%">
</p>

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

The developed system follows an image-to-text architecture.

<p align="center">
  <img src="docs/images/03-model-architecture.png" alt="Arabic Manuscript OCR model architecture" width="95%">
</p>

### Image Feature Extraction

CNN-based architectures were investigated for extracting visual features from manuscript images.

The project experimented with architectures including:

- MobileNet
- EfficientNet

The manuscript images were processed as **line images** rather than complete manuscript pages. This helped reduce memory usage and computational requirements during training.

### Transformer Decoder

The text-generation component uses a Transformer decoder to generate Arabic text from the extracted visual features.

The decoder includes:

- Token embeddings
- Positional information
- Causal self-attention
- Cross-attention to image features
- Feed-forward layers
- Autoregressive text generation

The cross-attention mechanism allows the decoder to use visual information from the manuscript while generating the output sequence.

---

## 📊 Dataset Preparation

Because a suitable ready-to-use labeled dataset for this specific task was not available for the project, a dataset was prepared for training.

The project focused on **manuscript line images** rather than complete pages.

The project aimed to collect approximately:

- **20,000 manuscript line images**
- **20,000 additional synthetic manuscript line images**

The dataset was then subjected to preprocessing and augmentation before training.

### Dataset Examples

<p align="center">
  <img src="docs/images/02-dataset-examples.png" alt="Collected and synthetic Arabic manuscript dataset examples" width="95%">
</p>

The resulting image representation used dimensions of approximately:

```text
70 × 800 × 3
```

Working with line images made it more practical to train the image-to-text model than using complete manuscript pages.

---

## 🧪 Model Experiments

Different architectures and configurations were investigated throughout the project.

### MobileNet Experiment

A MobileNet-based approach was explored as a visual feature extractor.

The project presentation reported the following result for one of the later MobileNet-based experiments:

```text
Training Accuracy:   99.66%
Validation Accuracy: 79.40%
```

The difference between training and validation performance demonstrated the difficulty of generalizing from the available training data.

### EfficientNet Experiment

A later experiment investigated EfficientNet as the visual feature extractor.

The project presentation reported:

```text
Training Accuracy:   99.82%
Validation Accuracy: 96.23%
```

This experiment showed a substantial improvement in validation performance compared with the reported MobileNet experiment.

> These values are the reported results of the corresponding experiments. They should not be interpreted as a universal OCR accuracy or as a standardized character-level OCR metric.

---

## 🔎 Attention Visualization

The implementation also includes attention visualization to inspect which regions of the manuscript image contribute to the generation of output tokens.

This provides an interpretable view of the relationship between the input image and generated text.

<p align="center">
  <img src="docs/images/04-attention-visualization.png" alt="Attention visualization for Arabic manuscript image-to-text generation" width="95%">
</p>

The process can be viewed conceptually as:

```text
Manuscript Image
       ↓
Visual Features
       ↓
Cross-Attention
       ↓
Generated Arabic Token
```

Attention visualization was used during the experiments to better understand the behavior of the image-to-text model.

---

## ⚠️ Limitations

The model is **not intended to recognize every style of Arabic handwriting**.

Its performance depends strongly on the handwriting styles represented in the training data.

Because the available dataset was limited in both **size and diversity**, the model generalized better to handwriting styles that were similar to those represented during training.

As a result, performance can decrease significantly when the model encounters substantially different handwriting styles or manuscript sources.

This limitation is one of the main challenges identified during the project.

### Additional limitations

- The dataset focused primarily on line-level images rather than complete manuscript pages.
- The diversity of historical handwriting styles was limited.
- The system is not a universal solution for all Arabic manuscripts.
- Traditional word-segmentation approaches were unreliable for some manuscript layouts.
- The project did not evaluate the system against a large standardized Arabic handwritten OCR benchmark.
- The reported accuracy values come from the experiments performed during the project and are not equivalent to CER or WER.

---

## 🚀 Future Improvements

Possible directions for improving the system include:

- Building a larger Arabic handwritten manuscript dataset
- Increasing the diversity of handwriting styles
- Adding more historical manuscript sources
- Improving generalization to unseen handwriting styles
- Developing more robust manuscript line and word segmentation
- Evaluating the system using OCR-specific metrics such as Character Error Rate (CER) and Word Error Rate (WER)
- Supporting page-level manuscript processing
- Exploring stronger vision-language architectures
- Expanding attention-based interpretability and analysis

---

## 🛠️ Technologies

- Python
- TensorFlow
- Keras
- OpenCV
- Convolutional Neural Networks
- Transformer
- Computer Vision
- Deep Learning
- Optical Character Recognition
- Arabic Handwriting Recognition

---

## 📂 Repository Structure

The repository contains experimental implementations developed during different stages of the project.

Some of the main files and directories include:

```text
arabic-manuscript-ocr/
│
├── docs/
│   └── images/
│       ├── 01-input-to-text.png
│       ├── 02-dataset-examples.png
│       ├── 03-model-architecture.png
│       └── 04-attention-visualization.png
│
├── images/
│
├── experiments/
│   ├── deep_learning/
│   │   ├── AAHR.ipynb
│   │   ├── image_captioning_EffNet.ipynb
│   │   └── image_to_text_transformer_training.py
│   │
│   └── traditional_image_processing/
│       ├── filter_comparison_experiment.py
│       ├── hanning_blur_experiment.py
│       ├── hough_line_detection_experiment.py
│       ├── mexican_hat_filter_experiment.py
│       ├── sift_feature_detection_experiment.py
│       ├── static_threshold_resize_experiment.py
│       └── thresholding_histogram_experiment.py
│
├── requirements.txt
│
├── .gitignore
└── README.md
```

These files represent different experiments and approaches explored during development, including traditional image processing and manuscript analysis.

---

## 🎓 Academic Context

**Project:** Arabic Manuscript Restoration and Digitization Using Deep Learning

**Institution:** University of Aleppo  
**Faculty:** Faculty of Informatics Engineering  
**Department:** Artificial Intelligence

The project was developed as an academic exploration of computer vision and deep learning techniques for the restoration, recognition, and digitization of Arabic manuscripts.

---

## 📌 Project Status

**Academic / Experimental Project**

This repository contains the implementation and experiments developed during the project.

It should be considered a research and learning project rather than a production-ready OCR system.

---

## 👨‍💻 Author

**Mahmoud Bannan**

Flutter Developer & Audio AI Engineer

[Portfolio](https://mahmoudbannan.com)
