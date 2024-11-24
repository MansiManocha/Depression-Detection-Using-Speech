# Depression-Detection-Using-Speech
Automated Depression Detector

Depression Detection System is a machine learning project designed to analyze speech data and detect depression using advanced techniques like **Convolutional Neural Networks (CNN)**, **Support Vector Machines (SVM)**, and **DistilBERT** (a transformer-based NLP model). The system leverages deep learning and traditional machine learning to classify speech/text data as **"Depressed"** or **"Non-Depressed"**.

---

🧠 **Features**

- **CNN for Speech Analysis**: Converts audio into spectrograms and classifies them using Convolutional Neural Networks.
- **SVM for Speech Analysis**: Uses TF-IDF feature extraction and SVM for classification.
- **DistilBERT**: Fine-tuned for text-based classification using transformer-based deep learning.
- **Multi-Model Integration**: Combines models to achieve robust results across various datasets.
- **Web Interface**: Includes a Flask-based web app for live testing.


## 📊 **Datasets**

- **DAIC-WOZ Depression Dataset**  
  - Includes audio recordings with transcriptions annotated for depression detection.  
  - Download [here](https://dcapswoz.ict.usc.edu/).


⚙️ **Installation**

1. **Clone the repository**:
   ```bash
   git clone https://github.com/username/depression-detector.git
   cd depression-detector
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate       # For Linux/macOS
   venv\Scripts\activate          # For Windows
   ```

3. **Download datasets**:
   Dataset is not included due to restrictions and huge size of dataset. You can request access to DAIC-WOZ Dataset from their official website.


**Training**

- Train the **CNN model**:
   ```bash
   Convolutional Neural Network.ipynb
   You can also use pretrained model directly- my_cnn_model.h5
   ```

- Train the **SVM model**:
   ```bash
   CNN-SVM.ipynb
   or You can also use pretrained model directly- my_svm.pkl
   ```

- Fine-tune **DistilBERT**:
   ```bash
  BERT.ipynb
   You can also use pretrained model directly- mybert.h5
   ```


**Web Application**
To launch the web app for live predictions:
```bash
cd app
python app.py
```
Visit `http://127.0.0.1:5000` in your browser.

---

🛠️ **Technologies Used**

- **Programming Language**: Python
- **Deep Learning Frameworks**: TensorFlow, PyTorch
- **Machine Learning Libraries**: Scikit-learn, Hugging Face Transformers
- **Frontend**: HTML, CSS (Flask Templates)
- **Visualization**: Matplotlib, Seaborn
- **Audio Processing**: LibROSA
- **Natural Language Processing**: Hugging Face, NLTK

---

📈 **Results**
![Screenshot 2024-11-13 220933](https://github.com/user-attachments/assets/5ccb1a7f-9733-4965-847e-972f6d90c56a)

![Screenshot 2024-11-13 220737](https://github.com/user-attachments/assets/d4ebb927-9fca-4b97-83ef-aa6d6520c502)

![Screenshot 2024-11-13 220855](https://github.com/user-attachments/assets/bd10fa0d-a84c-440f-880e-96e49db84555)




| Model        | Accuracy  |
|--------------|-----------|
| CNN          | 99.6%     |
| SVM          | 96.8%     |
| DistilBERT   | 98.5%     |

---

👩‍🔬 **Future Improvements**

- Integrating multimodal data (audio + text + emotions).
- Implementing attention mechanisms in CNNs.
- Expanding dataset with diverse speech and text samples.
- Deploying on cloud platforms for scalability.

---

🤝 **Contributing**

We welcome contributions!  
- Fork the repository.
- Create a feature branch.
- Submit a pull request with your changes.

---

✨ **Acknowledgements**

- [DAIC-WOZ Dataset](https://dcapswoz.ict.usc.edu/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [LibROSA](https://librosa.org/)
