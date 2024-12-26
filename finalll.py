import os
import warnings
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
import requests
import cv2

# Suppress warnings
warnings.filterwarnings("ignore")

# Define constants and configurations
IMAGE_SIZE = 225  # Resize all images to 225x225
BATCH_SIZE = 64
EPOCHS = 5
LABELS = {"Normal": 0, "Tumor": 1}

# Paths to datasets
SIAR_PATH = r"E:\data sets\DL\Siar-dataset"


# Build the CNN model
def build_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Dropout(0.5),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# Prepare an image for prediction
def prepare_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array


# Predict tumor presence
def predict_tumor(model, img_path):
    prepared_image = prepare_image(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    prediction = model.predict(prepared_image)
    prob = prediction[0][0]
    return f"Prediction: {'Tumor Detected' if prob > 0.5 else 'No Tumor Detected'}"


# Train the model (optional: only train if not already saved)
def train_model():
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,  # Normalize pixel values to [0,1]
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2  # 20% for validation
    )

    # Load images using flow_from_directory
    train_generator = train_datagen.flow_from_directory(
        SIAR_PATH,  # Path to the dataset directory
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',  # Binary classification (Tumor / Normal)
        subset='training'  # Use this subset for training
    )

    validation_generator = train_datagen.flow_from_directory(
        SIAR_PATH,  # Same directory for validation data
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',  # Binary classification
        subset='validation'  # Use this subset for validation
    )

    # Build the model
    model = build_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    model.summary()

    # Define callbacks
    callbacks = [
        ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5, min_lr=1e-5),
        EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    ]

    # Train the model using the generator
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Save the model
    model.save("tumor_detection_new1.keras")
    return model, history


# Load the model (if it exists, else train a new model)
def load_or_train_model():
    if os.path.exists("tumor_detection_new1.keras"):
        from tensorflow.keras.models import load_model
        model = load_model("tumor_detection_new1.keras")
        return model, None
    else:
        model, history = train_model()
        return model, history

#
# Streamlit interface for image upload and prediction
def run():
    # Sidebar Section
    st.markdown("""
            <style>
            .st-emotion-cache-h4xjwg{
            position:static;
            }
            .st-emotion-cache-13k62yr{
            display:flex;
            justify-content: center;
            align-items: center;
            background: url('https://png.pngtree.com/thumb_back/fw800/background/20240607/pngtree-digital-human-brain-with-blue-light-emitting-from-base-image_15861367.jpg') no-repeat;
            background-size: cover;
            background-position: center;
    }
            .st-emotion-cache-1dp5vir{
            visibility:hidden;
            }
            </style>
     """, unsafe_allow_html=True)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", ["Home", "App Info", "Visualizations","code"])

    if page == "Home":
        st.title("Tumor Detection")
        st.write("Upload an image to check if it contains a tumor.")

        # Image Upload Section
        choose = st.radio("",options=["Local Image","Url Image"])
        if choose == "Local Image":
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])
            if uploaded_file is not None:
                # Save the uploaded image temporarily
                img_path = f"temp_image.{uploaded_file.name.split('.')[-1]}"
                with open(img_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                # Load the model (or train if not exists)
                model, _ = load_or_train_model()
                # Predict the tumor presence
                with st.spinner('Making prediction...'):
                    result = predict_tumor(model, img_path)
                st.success('Prediction complete !')

                # Display the result
                st.image(uploaded_file, caption='Uploaded Image')
                st.write(result)
        elif choose == "Url Image":
            path = st.text_input("URL Image")
            if path:
                response = requests.get(path)
                img_path = "image.jpeg"
                with open(img_path, 'wb') as out_file:
                    for chunk in response.iter_content(1024):
                        out_file.write(chunk)
                uploaded_file = cv2.imread(img_path)
                uploaded_file = cv2.cvtColor(uploaded_file,cv2.COLOR_BGR2RGB)
                # Load the model (or train if not exists)
                model, _ = load_or_train_model()
                # Predict the tumor presence
                with st.spinner('Making prediction...'):
                    result = predict_tumor(model, img_path)
                st.success('Prediction complete !')

                # Display the result
                st.image(uploaded_file, caption='Uploaded Image')
                st.write(result)


        # Add a button to retrain the model
        if st.button('Retrain Model'):
            st.write("Training model... This might take a while.")
            model, _ = train_model()
            st.write("Model retrained successfully.")

    elif page == "App Info":
        st.title("About This App")
        st.write("""
            This application is designed for detecting tumors from brain scan images using a deep learning model.
            The model is trained using convolutional neural networks (CNNs) to achieve high accuracy.

            **Features:**
            - Upload a brain scan image and get predictions on tumor presence.
            - Visualize the training accuracy and loss (in the Visualizations section).
            - Retrain the model if required.
        """)

    elif page == "Visualizations":
        st.title("Visualizations")
        st.write("Here you can view important visualizations related to model training.")

        # Example visualization: Accuracy and Loss Graphs
        st.write("### Example Visualization: Model Performance")
        epochs = list(range(1, EPOCHS + 1))
        train_acc = [0.7 + 0.03 * i for i in range(EPOCHS)]  # Dummy data for demonstration
        val_acc = [0.65 + 0.025 * i for i in range(EPOCHS)]  # Dummy data for demonstration

        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_acc, label='Training Accuracy')
        plt.plot(epochs, val_acc, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy Over Epochs')
        plt.legend()
        st.pyplot(plt)

        st.write("### Example Visualization: Loss Curve")
        train_loss = [0.8 - 0.05 * i for i in range(EPOCHS)]  # Dummy data for demonstration
        val_loss = [0.85 - 0.04 * i for i in range(EPOCHS)]  # Dummy data for demonstration

        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Model Loss Over Epochs')
        plt.legend()
        st.pyplot(plt)
    elif page == "code":
        username = "project"
        password = "19111"
        st.markdown("# Login")
        with st.form("form1"):
            user = st.text_input("Username")
            passw = st.text_input("Password")
            s_state = st.form_submit_button("Submit")
        if s_state:
            if user == "" or passw == "":
                st.warning("Please Filled the form")
            elif user == username and passw == password:
                st.success("Submited Sucessfully")
                st.write("### Code")
                code ="""
import os
import warnings
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
import requests
import cv2

# Suppress warnings
warnings.filterwarnings("ignore")

# Define constants and configurations
IMAGE_SIZE = 225  # Resize all images to 225x225
BATCH_SIZE = 64
EPOCHS = 1
LABELS = {"Normal": 0, "Tumor": 1}

# Paths to datasets
SIAR_PATH = url_of_dataset


# Build the CNN model
def build_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Dropout(0.5),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# Prepare an image for prediction
def prepare_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array


# Predict tumor presence
def predict_tumor(model, img_path):
    prepared_image = prepare_image(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    prediction = model.predict(prepared_image)
    prob = prediction[0][0]
    return f"Prediction: {'Tumor Detected' if prob > 0.5 else 'No Tumor Detected'} with confidence {prob:.2f}"


# Train the model (optional: only train if not already saved)
def train_model():
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,  # Normalize pixel values to [0,1]
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2  # 20% for validation
    )

    # Load images using flow_from_directory
    train_generator = train_datagen.flow_from_directory(
        SIAR_PATH,  # Path to the dataset directory
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',  # Binary classification (Tumor / Normal)
        subset='training'  # Use this subset for training
    )

    validation_generator = train_datagen.flow_from_directory(
        SIAR_PATH,  # Same directory for validation data
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',  # Binary classification
        subset='validation'  # Use this subset for validation
    )

    # Build the model
    model = build_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    model.summary()

    # Define callbacks
    callbacks = [
        ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5, min_lr=1e-5),
        EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    ]

    # Train the model using the generator
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Save the model
    model.save("tumor_detection_new1.keras")
    return model, history


# Load the model (if it exists, else train a new model)
def load_or_train_model():
    if os.path.exists("tumor_detection_new1.keras"):
        from tensorflow.keras.models import load_model
        model = load_model("tumor_detection_new1.keras")
        return model, None
    else:
        model, history = train_model()
        return model, history
                """
                st.code(code,language="python")
            else:
                st.warning("Username or Password is Wrong")

if __name__ == "__main__":
    run()
