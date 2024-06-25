![9 -WearMaskNet-Real-Time-Face-Mask-Detection](https://github.com/VIS172/Face_Mask_Detection/assets/109724129/3ac76512-1ac0-4b49-a9bb-f014fa3bf01f)


Face Mask Detection using Convolutional Neural Networks (CNN)



This project implements a face mask detection system using a Convolutional Neural Network (CNN). The system can classify images of people into two categories: 'Mask' and 'No Mask'. This README provides a detailed guide on how to set up, train, and evaluate the face mask detection model.

DATASET: https://drive.google.com/drive/folders/1wJd6rPpzWxW2P1s2o9FnS6YOKGes7x_b?usp=sharing



Table of Contents
Installation
Dataset
Training the Model
Evaluating the Model
Using the Model
Directory Structure




Installation
To set up the environment for this project, you need Python 3.6 or above. Follow these steps to install the required packages:

Clone the Repository:

sh
Copy code
git clone https://github.com/your-username/face-mask-detection.git
cd face-mask-detection
Create a Virtual Environment:

sh
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install Dependencies:

sh
Copy code
pip install -r requirements.txt
Dataset
The dataset used for training and evaluation should contain images labeled as 'Mask' and 'No Mask'. You can download a pre-labeled dataset or create your own. Make sure the dataset is structured as follows:

css
Copy code
dataset/
    ├── train/
    │   ├── Mask/
    │   └── NoMask/
    └── test/
        ├── Mask/
        └── NoMask/
Training the Model
To train the CNN model, run the train.py script:

sh
Copy code
python train.py
The script will:

Load and preprocess the dataset.
Build the CNN model architecture.
Train the model on the training dataset.
Save the trained model to the models directory.
Training Parameters
You can adjust the training parameters such as batch size, number of epochs, learning rate, etc., in the config.py file.

Evaluating the Model
To evaluate the trained model on the test dataset, run the evaluate.py script:

sh
Copy code
python evaluate.py
This script will:

Load the saved model.
Evaluate the model on the test dataset.
Print the evaluation metrics (accuracy, precision, recall, F1 score).
Using the Model
You can use the trained model to make predictions on new images. Run the predict.py script:

sh
Copy code
python predict.py --image_path path/to/image.jpg
This script will:

Load the saved model.
Load and preprocess the input image.
Predict whether the person in the image is wearing a mask or not.
Output the prediction result.
Directory Structure
The project directory is structured as follows:

arduino
Copy code
face-mask-detection/
    ├── dataset/
    │   ├── train/
    │   └── test/
    ├── models/
    │   └── face_mask_detector.model
    ├── scripts/
    │   ├── train.py
    │   ├── evaluate.py
    │   └── predict.py
    ├── utils/
    │   ├── data_preprocessing.py
    │   └── model_builder.py
    ├── config.py
    ├── requirements.txt
    └── README.md
dataset/: Directory containing the training and testing datasets.
models/: Directory where the trained model is saved.
scripts/: Directory containing the main scripts for training, evaluation, and prediction.
utils/: Directory containing utility scripts for data preprocessing and model building.
config.py: Configuration file for setting training parameters.
requirements.txt: File listing the required Python packages.
README.md: Project documentation file.
config.py: Configuration file for setting training parameters.
requirements.txt: File listing the required Python packages.
README.md: Project documentation file.
Acknowledgements
This project is inspired by the need for automatic face mask detection to ensure public health safety. The dataset and initial model architecture are based on publicly available resources and research papers in the field of computer vision and deep learning.
