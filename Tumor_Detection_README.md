
# Tumor Detection Using Deep Learning

This project is a deep learning-based tumor detection system using Convolutional Neural Networks (CNNs). The model is trained on brain scan images to classify whether an image contains a tumor or not. The application is built using Streamlit for the user interface and TensorFlow for the model.

## Requirements

The following Python libraries are required to run the project:

- `tensorflow`
- `streamlit`
- `numpy`
- `opencv-python`
- `matplotlib`
- `requests`

You can install the required libraries using pip:

```bash
pip install tensorflow streamlit numpy opencv-python matplotlib requests
```

## File Structure

```
.
├── app.py                  # Main Streamlit application script
├── tumor_detection_new1.keras  # Pre-trained model (if exists)
└── README.md               # Project overview and setup instructions
```

## Model Architecture

The model is a Convolutional Neural Network (CNN) with the following layers:

1. Conv2D (32 filters, 3x3 kernel, ReLU activation)
2. MaxPooling2D (2x2 pool size)
3. Dropout (25%)
4. Conv2D (64 filters, 3x3 kernel, ReLU activation)
5. MaxPooling2D (2x2 pool size)
6. Dropout (25%)
7. Conv2D (128 filters, 3x3 kernel, ReLU activation)
8. MaxPooling2D (2x2 pool size)
9. Dropout (50%)
10. Flatten
11. Dense (128 neurons, ReLU activation)
12. Dropout (50%)
13. Dense (1 neuron, Sigmoid activation)

The model is compiled using the Adam optimizer and binary crossentropy loss function.

## Training

The model is trained using a dataset of brain scan images. You can train the model by running the following code in the `app.py`:

```python
model, history = train_model()
```

If the model has already been trained and saved as `tumor_detection_new1.keras`, the application will load the saved model.

## Prediction

Once the model is trained, you can upload an image for prediction. The application will predict whether the image contains a tumor or not. The prediction is based on the confidence score produced by the model.

### Image Upload

You can upload images in the following formats: JPG, JPEG, PNG, or BMP. The model will predict whether the uploaded image contains a tumor or not.

### URL Image

Alternatively, you can provide a URL link to an image. The application will fetch the image and make a prediction based on the image provided.

## Streamlit Interface

The Streamlit app provides a simple interface for interaction:

- **Home**: Upload an image to check for tumor presence.
- **App Info**: Learn more about the app.
- **Visualizations**: View model training performance (accuracy, loss graphs).
- **Code**: View the source code of the app after logging in.

## Retraining the Model

You can retrain the model by clicking on the "Retrain Model" button in the app. This will initiate the training process and save the new model once training is complete.

## License

This project is licensed under the MIT License.
