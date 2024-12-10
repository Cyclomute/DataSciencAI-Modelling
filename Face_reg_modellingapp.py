{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "a124249b-6d83-4a0c-99e1-c8c6d2a2d57f",
   "metadata": {},
   "outputs": [
    {
     "ename": "FileNotFoundError",
     "evalue": "[Errno 2] Unable to open file (unable to open file: name = 'facial_recognition_model_final1.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mFileNotFoundError\u001b[0m                         Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[28], line 12\u001b[0m\n\u001b[1;32m      8\u001b[0m \u001b[38;5;129m@st\u001b[39m\u001b[38;5;241m.\u001b[39mcache_resource\n\u001b[1;32m      9\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mload_model\u001b[39m():\n\u001b[1;32m     10\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m tf\u001b[38;5;241m.\u001b[39mkeras\u001b[38;5;241m.\u001b[39mmodels\u001b[38;5;241m.\u001b[39mload_model(\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mfacial_recognition_model_final1.h5\u001b[39m\u001b[38;5;124m'\u001b[39m)\n\u001b[0;32m---> 12\u001b[0m model \u001b[38;5;241m=\u001b[39m load_model()\n\u001b[1;32m     14\u001b[0m \u001b[38;5;66;03m# Preprocess the image\u001b[39;00m\n\u001b[1;32m     15\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mpreprocess_image\u001b[39m(image):\n\u001b[1;32m     16\u001b[0m     \u001b[38;5;66;03m# Convert to RGB if uploaded as BGR or grayscale\u001b[39;00m\n",
      "File \u001b[0;32m/opt/anaconda3/lib/python3.12/site-packages/streamlit/runtime/caching/cache_utils.py:210\u001b[0m, in \u001b[0;36mmake_cached_func_wrapper.<locals>.wrapper\u001b[0;34m(*args, **kwargs)\u001b[0m\n\u001b[1;32m    208\u001b[0m \u001b[38;5;129m@functools\u001b[39m\u001b[38;5;241m.\u001b[39mwraps(info\u001b[38;5;241m.\u001b[39mfunc)\n\u001b[1;32m    209\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mwrapper\u001b[39m(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs):\n\u001b[0;32m--> 210\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m cached_func(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n",
      "File \u001b[0;32m/opt/anaconda3/lib/python3.12/site-packages/streamlit/runtime/caching/cache_utils.py:239\u001b[0m, in \u001b[0;36mCachedFunc.__call__\u001b[0;34m(self, *args, **kwargs)\u001b[0m\n\u001b[1;32m    237\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_info\u001b[38;5;241m.\u001b[39mshow_spinner \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;28misinstance\u001b[39m(\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_info\u001b[38;5;241m.\u001b[39mshow_spinner, \u001b[38;5;28mstr\u001b[39m):\n\u001b[1;32m    238\u001b[0m     \u001b[38;5;28;01mwith\u001b[39;00m spinner(message, _cache\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mTrue\u001b[39;00m):\n\u001b[0;32m--> 239\u001b[0m         \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_get_or_create_cached_value(args, kwargs)\n\u001b[1;32m    240\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[1;32m    241\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_get_or_create_cached_value(args, kwargs)\n",
      "File \u001b[0;32m/opt/anaconda3/lib/python3.12/site-packages/streamlit/runtime/caching/cache_utils.py:266\u001b[0m, in \u001b[0;36mCachedFunc._get_or_create_cached_value\u001b[0;34m(self, func_args, func_kwargs)\u001b[0m\n\u001b[1;32m    264\u001b[0m \u001b[38;5;28;01mexcept\u001b[39;00m CacheKeyNotFoundError:\n\u001b[1;32m    265\u001b[0m     \u001b[38;5;28;01mpass\u001b[39;00m\n\u001b[0;32m--> 266\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_handle_cache_miss(cache, value_key, func_args, func_kwargs)\n",
      "File \u001b[0;32m/opt/anaconda3/lib/python3.12/site-packages/streamlit/runtime/caching/cache_utils.py:322\u001b[0m, in \u001b[0;36mCachedFunc._handle_cache_miss\u001b[0;34m(self, cache, value_key, func_args, func_kwargs)\u001b[0m\n\u001b[1;32m    318\u001b[0m \u001b[38;5;66;03m# We acquired the lock before any other thread. Compute the value!\u001b[39;00m\n\u001b[1;32m    319\u001b[0m \u001b[38;5;28;01mwith\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_info\u001b[38;5;241m.\u001b[39mcached_message_replay_ctx\u001b[38;5;241m.\u001b[39mcalling_cached_function(\n\u001b[1;32m    320\u001b[0m     \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_info\u001b[38;5;241m.\u001b[39mfunc, \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_info\u001b[38;5;241m.\u001b[39mallow_widgets\n\u001b[1;32m    321\u001b[0m ):\n\u001b[0;32m--> 322\u001b[0m     computed_value \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_info\u001b[38;5;241m.\u001b[39mfunc(\u001b[38;5;241m*\u001b[39mfunc_args, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mfunc_kwargs)\n\u001b[1;32m    324\u001b[0m \u001b[38;5;66;03m# We've computed our value, and now we need to write it back to the cache\u001b[39;00m\n\u001b[1;32m    325\u001b[0m \u001b[38;5;66;03m# along with any \"replay messages\" that were generated during value computation.\u001b[39;00m\n\u001b[1;32m    326\u001b[0m messages \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_info\u001b[38;5;241m.\u001b[39mcached_message_replay_ctx\u001b[38;5;241m.\u001b[39m_most_recent_messages\n",
      "Cell \u001b[0;32mIn[28], line 10\u001b[0m, in \u001b[0;36mload_model\u001b[0;34m()\u001b[0m\n\u001b[1;32m      8\u001b[0m \u001b[38;5;129m@st\u001b[39m\u001b[38;5;241m.\u001b[39mcache_resource\n\u001b[1;32m      9\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mload_model\u001b[39m():\n\u001b[0;32m---> 10\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m tf\u001b[38;5;241m.\u001b[39mkeras\u001b[38;5;241m.\u001b[39mmodels\u001b[38;5;241m.\u001b[39mload_model(\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mfacial_recognition_model_final1.h5\u001b[39m\u001b[38;5;124m'\u001b[39m)\n",
      "File \u001b[0;32m/opt/anaconda3/lib/python3.12/site-packages/keras/src/saving/saving_api.py:194\u001b[0m, in \u001b[0;36mload_model\u001b[0;34m(filepath, custom_objects, compile, safe_mode)\u001b[0m\n\u001b[1;32m    187\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m saving_lib\u001b[38;5;241m.\u001b[39mload_model(\n\u001b[1;32m    188\u001b[0m         filepath,\n\u001b[1;32m    189\u001b[0m         custom_objects\u001b[38;5;241m=\u001b[39mcustom_objects,\n\u001b[1;32m    190\u001b[0m         \u001b[38;5;28mcompile\u001b[39m\u001b[38;5;241m=\u001b[39m\u001b[38;5;28mcompile\u001b[39m,\n\u001b[1;32m    191\u001b[0m         safe_mode\u001b[38;5;241m=\u001b[39msafe_mode,\n\u001b[1;32m    192\u001b[0m     )\n\u001b[1;32m    193\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mstr\u001b[39m(filepath)\u001b[38;5;241m.\u001b[39mendswith((\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m.h5\u001b[39m\u001b[38;5;124m\"\u001b[39m, \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m.hdf5\u001b[39m\u001b[38;5;124m\"\u001b[39m)):\n\u001b[0;32m--> 194\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m legacy_h5_format\u001b[38;5;241m.\u001b[39mload_model_from_hdf5(\n\u001b[1;32m    195\u001b[0m         filepath, custom_objects\u001b[38;5;241m=\u001b[39mcustom_objects, \u001b[38;5;28mcompile\u001b[39m\u001b[38;5;241m=\u001b[39m\u001b[38;5;28mcompile\u001b[39m\n\u001b[1;32m    196\u001b[0m     )\n\u001b[1;32m    197\u001b[0m \u001b[38;5;28;01melif\u001b[39;00m \u001b[38;5;28mstr\u001b[39m(filepath)\u001b[38;5;241m.\u001b[39mendswith(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m.keras\u001b[39m\u001b[38;5;124m\"\u001b[39m):\n\u001b[1;32m    198\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mValueError\u001b[39;00m(\n\u001b[1;32m    199\u001b[0m         \u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mFile not found: filepath=\u001b[39m\u001b[38;5;132;01m{\u001b[39;00mfilepath\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m. \u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m    200\u001b[0m         \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mPlease ensure the file is an accessible `.keras` \u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m    201\u001b[0m         \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mzip file.\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m    202\u001b[0m     )\n",
      "File \u001b[0;32m/opt/anaconda3/lib/python3.12/site-packages/keras/src/legacy/saving/legacy_h5_format.py:116\u001b[0m, in \u001b[0;36mload_model_from_hdf5\u001b[0;34m(filepath, custom_objects, compile)\u001b[0m\n\u001b[1;32m    114\u001b[0m opened_new_file \u001b[38;5;241m=\u001b[39m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28misinstance\u001b[39m(filepath, h5py\u001b[38;5;241m.\u001b[39mFile)\n\u001b[1;32m    115\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m opened_new_file:\n\u001b[0;32m--> 116\u001b[0m     f \u001b[38;5;241m=\u001b[39m h5py\u001b[38;5;241m.\u001b[39mFile(filepath, mode\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mr\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[1;32m    117\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[1;32m    118\u001b[0m     f \u001b[38;5;241m=\u001b[39m filepath\n",
      "File \u001b[0;32m/opt/anaconda3/lib/python3.12/site-packages/h5py/_hl/files.py:562\u001b[0m, in \u001b[0;36mFile.__init__\u001b[0;34m(self, name, mode, driver, libver, userblock_size, swmr, rdcc_nslots, rdcc_nbytes, rdcc_w0, track_order, fs_strategy, fs_persist, fs_threshold, fs_page_size, page_buf_size, min_meta_keep, min_raw_keep, locking, alignment_threshold, alignment_interval, meta_block_size, **kwds)\u001b[0m\n\u001b[1;32m    553\u001b[0m     fapl \u001b[38;5;241m=\u001b[39m make_fapl(driver, libver, rdcc_nslots, rdcc_nbytes, rdcc_w0,\n\u001b[1;32m    554\u001b[0m                      locking, page_buf_size, min_meta_keep, min_raw_keep,\n\u001b[1;32m    555\u001b[0m                      alignment_threshold\u001b[38;5;241m=\u001b[39malignment_threshold,\n\u001b[1;32m    556\u001b[0m                      alignment_interval\u001b[38;5;241m=\u001b[39malignment_interval,\n\u001b[1;32m    557\u001b[0m                      meta_block_size\u001b[38;5;241m=\u001b[39mmeta_block_size,\n\u001b[1;32m    558\u001b[0m                      \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwds)\n\u001b[1;32m    559\u001b[0m     fcpl \u001b[38;5;241m=\u001b[39m make_fcpl(track_order\u001b[38;5;241m=\u001b[39mtrack_order, fs_strategy\u001b[38;5;241m=\u001b[39mfs_strategy,\n\u001b[1;32m    560\u001b[0m                      fs_persist\u001b[38;5;241m=\u001b[39mfs_persist, fs_threshold\u001b[38;5;241m=\u001b[39mfs_threshold,\n\u001b[1;32m    561\u001b[0m                      fs_page_size\u001b[38;5;241m=\u001b[39mfs_page_size)\n\u001b[0;32m--> 562\u001b[0m     fid \u001b[38;5;241m=\u001b[39m make_fid(name, mode, userblock_size, fapl, fcpl, swmr\u001b[38;5;241m=\u001b[39mswmr)\n\u001b[1;32m    564\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28misinstance\u001b[39m(libver, \u001b[38;5;28mtuple\u001b[39m):\n\u001b[1;32m    565\u001b[0m     \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_libver \u001b[38;5;241m=\u001b[39m libver\n",
      "File \u001b[0;32m/opt/anaconda3/lib/python3.12/site-packages/h5py/_hl/files.py:235\u001b[0m, in \u001b[0;36mmake_fid\u001b[0;34m(name, mode, userblock_size, fapl, fcpl, swmr)\u001b[0m\n\u001b[1;32m    233\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m swmr \u001b[38;5;129;01mand\u001b[39;00m swmr_support:\n\u001b[1;32m    234\u001b[0m         flags \u001b[38;5;241m|\u001b[39m\u001b[38;5;241m=\u001b[39m h5f\u001b[38;5;241m.\u001b[39mACC_SWMR_READ\n\u001b[0;32m--> 235\u001b[0m     fid \u001b[38;5;241m=\u001b[39m h5f\u001b[38;5;241m.\u001b[39mopen(name, flags, fapl\u001b[38;5;241m=\u001b[39mfapl)\n\u001b[1;32m    236\u001b[0m \u001b[38;5;28;01melif\u001b[39;00m mode \u001b[38;5;241m==\u001b[39m \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mr+\u001b[39m\u001b[38;5;124m'\u001b[39m:\n\u001b[1;32m    237\u001b[0m     fid \u001b[38;5;241m=\u001b[39m h5f\u001b[38;5;241m.\u001b[39mopen(name, h5f\u001b[38;5;241m.\u001b[39mACC_RDWR, fapl\u001b[38;5;241m=\u001b[39mfapl)\n",
      "File \u001b[0;32mh5py/_objects.pyx:54\u001b[0m, in \u001b[0;36mh5py._objects.with_phil.wrapper\u001b[0;34m()\u001b[0m\n",
      "File \u001b[0;32mh5py/_objects.pyx:55\u001b[0m, in \u001b[0;36mh5py._objects.with_phil.wrapper\u001b[0;34m()\u001b[0m\n",
      "File \u001b[0;32mh5py/h5f.pyx:102\u001b[0m, in \u001b[0;36mh5py.h5f.open\u001b[0;34m()\u001b[0m\n",
      "\u001b[0;31mFileNotFoundError\u001b[0m: [Errno 2] Unable to open file (unable to open file: name = 'facial_recognition_model_final1.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)"
     ]
    }
   ],
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
    "    return tf.keras.models.load_model('facial_recognition_model_final1.h5')\n",
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
   "execution_count": 19,
   "id": "688a7235-fba0-49b4-a1df-49d5ee28126c",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8545cf05-12cd-4d08-977d-b2b8faab082c",
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
