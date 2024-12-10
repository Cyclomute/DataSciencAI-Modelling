{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "a124249b-6d83-4a0c-99e1-c8c6d2a2d57f",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "from PIL import Image\n",
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "import cv2\n",
    "\n",
    "# Load the trained model\n",
    "@st.cache_resource\n",
    "def load_model():\n",
    "    return tf.keras.models.load_model('/Users/morrisonosiezagha/Downloads/facial_recognition_model_final1.keras')\n",
    "\n",
    "model = load_model()\n",
    "\n",
    "# Preprocess the image\n",
    "def preprocess_image(image):\n",
    "    # Convert to RGB if uploaded as BGR or grayscale\n",
    "    image = image.convert(\"RGB\")\n",
    "    # Resize to the model's expected input size\n",
    "    img_resized = image.resize((224, 224))  # Change to your model's input size\n",
    "    # Normalize pixel values (if required by the model)\n",
    "    img_array = np.array(img_resized) / 255.0  # Scale to 0-1\n",
    "    # Add batch dimension\n",
    "    return np.expand_dims(img_array, axis=0)\n",
    "\n",
    "# Face recognition function\n",
    "def recognize_face(image):\n",
    "    processed_img = preprocess_image(image)\n",
    "    prediction = model.predict(processed_img)\n",
    "    return prediction  # Adjust this to your model's output structure\n",
    "\n",
    "# Streamlit app\n",
    "st.title(\"Face Recognition App\")\n",
    "st.write(\"Upload an image to identify faces.\")\n",
    "\n",
    "uploaded_file = st.file_uploader(\"Choose an image...\", type=[\"jpg\", \"jpeg\", \"png\"])\n",
    "\n",
    "if uploaded_file is not None:\n",
    "    # Display the uploaded image\n",
    "    image = Image.open(uploaded_file)\n",
    "    st.image(image, caption=\"Uploaded Image\", use_column_width=True)\n",
    "    \n",
    "    # Perform face recognition\n",
    "    st.write(\"Processing...\")\n",
    "    predictions = recognize_face(image)\n",
    "    \n",
    "    # Display results\n",
    "    st.write(\"Recognition Results:\")\n",
    "    st.write(predictions)  # Customize to show human-readable output\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "688a7235-fba0-49b4-a1df-49d5ee28126c",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import numpy as np\n",
    "from PIL import Image\n",
    "import tensorflow as tf\n",
    "\n",
    "# Load the pre-trained model\n",
    "@st.cache_resource\n",
    "def load_model():\n",
    "    return tf.keras.models.load_model('/Users/morrisonosiezagha/Downloads/facial_recognition_model_final1.keras')\n",
    "\n",
    "model = load_model()\n",
    "\n",
    "# Preprocess image\n",
    "def preprocess_image(image):\n",
    "    image = image.resize((224, 224))  # Adjust size based on model input\n",
    "    image_array = np.array(image) / 255.0  # Normalize\n",
    "    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension\n",
    "    return image_array\n",
    "\n",
    "# App UI\n",
    "st.title(\"Face Recognition App\")\n",
    "uploaded_file = st.file_uploader(\"Upload an image\", type=[\"jpg\", \"jpeg\", \"png\"])\n",
    "\n",
    "if uploaded_file is not None:\n",
    "    image = Image.open(uploaded_file)\n",
    "    st.image(image, caption=\"Uploaded Image\", use_column_width=True)\n",
    "    preprocessed_image = preprocess_image(image)\n",
    "    predictions = model.predict(preprocessed_image)\n",
    "    \n",
    "    # Interpret and display results\n",
    "    class_names = [\"Person A\", \"Person B\", \"Person C\"]  # Replace with your classes\n",
    "    predicted_class = class_names[np.argmax(predictions)]\n",
    "    confidence = np.max(predictions) * 100\n",
    "    \n",
    "    st.write(f\"Prediction: {predicted_class}\")\n",
    "    st.write(f\"Confidence: {confidence:.2f}%\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "8545cf05-12cd-4d08-977d-b2b8faab082c",
   "metadata": {},
   "outputs": [
    {
     "ename": "SyntaxError",
     "evalue": "invalid syntax (1722003430.py, line 1)",
     "output_type": "error",
     "traceback": [
      "\u001b[0;36m  Cell \u001b[0;32mIn[11], line 1\u001b[0;36m\u001b[0m\n\u001b[0;31m    streamlit run /opt/anaconda3/lib/python3.12/site-packages/ipykernel_launcher.py [ARGUMENTS]\u001b[0m\n\u001b[0m              ^\u001b[0m\n\u001b[0;31mSyntaxError\u001b[0m\u001b[0;31m:\u001b[0m invalid syntax\n"
     ]
    }
   ],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ad9fc533-8bee-4b90-a492-6fa3c2300587",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
