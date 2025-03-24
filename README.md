# Computer Vision Learning Projects

This repository documents my journey learning computer vision concepts and applications through practical projects. Each project explores different aspects of computer vision using OpenCV, mediapipe, and other related libraries.

## Project Overview

The repository contains multiple mini-projects focused on various computer vision applications including:

- Blob detection
- Color detection
- Face detection and blurring
- Text recognition (OCR)
- Parking space detection

Each project demonstrates different techniques and libraries to solve specific computer vision problems.

## Project Structure

```
ComputerVision/
├── BlobDetection/        # Detects blobs in images
├── ColorDetection/       # Detects specific colors in video feed
├── FaceBlur/             # Detects and blurs faces in images and video
├── Image Detection/      # Parking space detection application
├── TextDetection/        # OCR implementation using EasyOCR and Tesseract
├── ObjectDetection/      # SAM2 object detection inferance using webcam
```

## Installation

1. Clone this repository
   ```
   git clone https://github.com/theCarterDavis/ComputerVision.git
   cd ComputerVision
   ```

2. Create and activate a virtual environment (recommended)
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies
   ```
   pip install -r requirements.txt
   ```

## Project Descriptions

### Blob Detection
Detects blob-like structures in images using OpenCV's SimpleBlobDetector with customizable parameters for thresholds, area, circularity, convexity, and inertia.

### Color Detection
Real-time color detection using webcam feed. The application converts BGR color space to HSV for more robust color detection and draws bounding boxes around detected colors.

### Face Blur
Uses mediapipe's face detection model to identify faces in images or webcam feed and applies Gaussian blur to anonymize them. Includes both static image processing and real-time webcam implementation.

### Parking Space Detection
A computer vision application that:
- Uses connected components to identify parking spots from a mask
- Analyzes video feed to determine if parking spaces are occupied
- Keeps track of available parking spaces
- Displays a count of available spots

### Text Detection
Implements Optical Character Recognition (OCR) using two different libraries:
- EasyOCR for scene text detection
- Pytesseract for document text recognition

### Object Detection(In progress)
Implements object segmentation through the use of SAM2:
- Goal is to detect shapes and compare them to a key to test the accuracy for different shapes
- Currently will allow the user to click on any point in the webcam feed and run SAM2's inference to show what it views as the object clicked on

## Usage Examples

### Face Blur Example
```python
python FaceBlur/fbMain.py  # For single image blur
python FaceBlur/wcBlur.py  # For webcam real-time face blurring
```

### Color Detection Example
```python
python ColorDetection/cdMain.py
```

### Running the Parking Space Counter
```python
python Image\ Detection/MainID.py
```

## Dependencies

Key libraries used in these projects:
- OpenCV (cv2): Core computer vision library
- Mediapipe: For face detection
- NumPy: For numerical operations
- EasyOCR & Pytesseract: For text recognition
- PIL/Pillow: For image processing
- Scikit-image: For image transformation
- Ultralytics: For instance segmentation

## Learning Resources

Some helpful resources for learning computer vision:
- [OpenCV Documentation](https://docs.opencv.org/)
- [PyImageSearch Tutorials](https://pyimagesearch.com/)
- [Learn OpenCV Github/Blog](https://github.com/spmallick/learnopencv)
- [Open CV Python Tutorials Repository](https://github.com/niconielsen32/opencv-python-tutorials)

## Future Projects

- 3D Computer Vision
