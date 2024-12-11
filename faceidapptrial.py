{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "c1e8a5cf-3b19-4e71-8775-2594959a8c0d",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "from PIL import Image\n",
    "import numpy as np\n",
    "import tensorflow as tf\n",
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
   "execution_count": None,
   "id": "17819f16-a528-4f12-b871-24a2143bcc4f",
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
