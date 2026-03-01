{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "23a1c40a-99b7-4836-90d2-f118d50de8d5",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.\n",
      "2026-03-01 05:43:29.391 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-03-01 05:43:29.746 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\sarat\\venvs\\ml_env\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2026-03-01 05:43:29.747 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-03-01 05:43:29.747 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-03-01 05:43:29.748 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-03-01 05:43:29.749 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-03-01 05:43:29.749 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-03-01 05:43:29.750 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-03-01 05:43:29.750 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-03-01 05:43:29.751 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-03-01 05:43:29.751 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-03-01 05:43:29.752 Session state does not function when running a script without `streamlit run`\n",
      "2026-03-01 05:43:29.753 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-03-01 05:43:29.754 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-03-01 05:43:29.755 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-03-01 05:43:29.758 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-03-01 05:43:29.759 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-03-01 05:43:29.760 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-03-01 05:43:29.761 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-03-01 05:43:29.762 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-03-01 05:43:29.763 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-03-01 05:43:29.763 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-03-01 05:43:29.764 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-03-01 05:43:29.764 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "# Step 1: Import Libraries and Load the Model\n",
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "from tensorflow.keras.datasets import imdb\n",
    "from tensorflow.keras.preprocessing import sequence\n",
    "from tensorflow.keras.models import load_model\n",
    "\n",
    "# Load the IMDB dataset word index\n",
    "word_index = imdb.get_word_index()\n",
    "reverse_word_index = {value: key for key, value in word_index.items()}\n",
    "\n",
    "# Load the pre-trained model with ReLU activation\n",
    "model = load_model('simple_rnn_imdb.h5')\n",
    "\n",
    "# Step 2: Helper Functions\n",
    "# Function to decode reviews\n",
    "def decode_review(encoded_review):\n",
    "    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])\n",
    "\n",
    "# Function to preprocess user input\n",
    "def preprocess_text(text):\n",
    "    words = text.lower().split()\n",
    "    encoded_review = [word_index.get(word, 2) + 3 for word in words]\n",
    "    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)\n",
    "    return padded_review\n",
    "\n",
    "\n",
    "import streamlit as st\n",
    "## streamlit app\n",
    "# Streamlit app\n",
    "st.title('IMDB Movie Review Sentiment Analysis')\n",
    "st.write('Enter a movie review to classify it as positive or negative.')\n",
    "\n",
    "# User input\n",
    "user_input = st.text_area('Movie Review')\n",
    "\n",
    "if st.button('Classify'):\n",
    "\n",
    "    preprocessed_input=preprocess_text(user_input)\n",
    "\n",
    "    ## MAke prediction\n",
    "    prediction=model.predict(preprocessed_input)\n",
    "    sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'\n",
    "\n",
    "    # Display the result\n",
    "    st.write(f'Sentiment: {sentiment}')\n",
    "    st.write(f'Prediction Score: {prediction[0][0]}')\n",
    "else:\n",
    "    st.write('Please enter a movie review.')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "805308a7-9378-49a0-9c12-abdd01afb85d",
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
   "version": "3.12.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
