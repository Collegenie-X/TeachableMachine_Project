# Mask Detection System User Guide

This document explains how to run the mask detection model created with Teachable Machine in a local environment.

## 1. Prerequisites

### Install Required Packages
```bash
pip install -r requirements.txt
```

### Prepare Teachable Machine Model
1. Create an image project on [Teachable Machine](https://teachablemachine.withgoogle.com/)
2. Train the model with mask wearing/not wearing images
3. Select "Export Model" > "TensorFlow" > Download "Keras"
4. Extract the downloaded zip file

## 2. Prepare Model Files

Save the downloaded model files to the following location:

```
models/converted_keras/keras_model.h5
models/converted_keras/labels.txt
```

Or extract with the following structure:
```
models/
└── converted_keras/
    ├── keras_model.h5
    └── labels.txt
```

## 3. Run the Application

### Run Mask Detection Application
```bash
cd src
python mask_detection_app.py
```

### Command Line Options:
```
--model: Model file path (default: ../models/converted_keras/keras_model.h5)
--labels: Labels file path (default: ../models/converted_keras/labels.txt)
--camera: Camera device number (default: 0)
--image_size: Input image size (default: 224)
--quiet: Run with minimal output (reduces warning messages)
```

Examples:
```bash
# Standard execution
python mask_detection_app.py

# Run in quiet mode (no warning messages)
python mask_detection_app.py --quiet

# Use a different camera device
python mask_detection_app.py --camera 1

# Change image size
python mask_detection_app.py --image_size 96
```

### Test with a Single Image
To test with a single image, you can use the test script:

```bash
cd src
python test_model.py --image ../path/to/your/image.jpg
```

## 4. How to Use

- When the application runs, it detects faces through the webcam and determines if a mask is being worn.
- Faces with masks are highlighted with a green border.
- Faces without masks are highlighted with a red border and a warning message.
- Press 'q' to quit.

## 5. Troubleshooting

### Model Loading Errors and Warning Messages
Recent TensorFlow versions may have compatibility issues when loading Teachable Machine models. This code addresses these issues in the following ways:

1. DepthwiseConv2D Layer Patch:
   - A layer patch is applied before loading the model to fix the `groups` parameter issue.

2. Warning Message Filtering:
   - The following code has been added to hide unnecessary warning messages:
   ```python
   warnings.filterwarnings('ignore', category=UserWarning)
   os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
   ```

3. Quiet Mode:
   - Use the `--quiet` option to minimize unnecessary output.

### Camera Errors
If you cannot open the camera, check the following:
- Make sure the camera is not being used by another application
- Check if the camera device number is correct (try a different number)
- Verify that the camera drivers are properly installed 