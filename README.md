![9 -WearMaskNet-Real-Time-Face-Mask-Detection](https://github.com/VIS172/Face_Mask_Detection/assets/109724129/3ac76512-1ac0-4b49-a9bb-f014fa3bf01f)


Face Mask Detection using Convolutional Neural Networks (CNN)



This project implements a face mask detection system using a Convolutional Neural Network (CNN). The system can classify images of people into two categories: 'Mask' and 'No Mask'. This README provides a detailed guide on how to set up, train, and evaluate the face mask detection model.

DATASET: https://drive.google.com/drive/folders/1wJd6rPpzWxW2P1s2o9FnS6YOKGes7x_b?usp=sharing


## Table of Contents
1. [Installation](#installation)
2. [Dataset](#dataset)
3. [Training the Model](#training-the-model)
4. [Evaluating the Model](#evaluating-the-model)
5. [Using the Model](#using-the-model)
6. [Directory Structure](#directory-structure)
7. [Acknowledgements](#acknowledgements)

## Installation

To set up the environment for this project, you need Python 3.6 or above. Follow these steps to install the required packages:

1. **Clone the Repository:**
    ```sh
    git clone https://github.com/your-username/face-mask-detection.git
    cd face-mask-detection
    ```

2. **Create a Virtual Environment:**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install Dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

## Dataset

The dataset used for training and evaluation should contain images labeled as 'Mask' and 'No Mask'. You can download a pre-labeled dataset or create your own. Make sure the dataset is structured as follows:


## Training the Model

To train the CNN model, run the `train.py` script:

```sh
python train.py
python evaluate.py
python predict.py --image_path path/to/image.jpg
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






