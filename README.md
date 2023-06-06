# Leaf-Disease-Classification
Leaf Disease Classification using Deep Learning, Tensorflow, and deploying on a web based app

The proposed methodology aims to achieve accurate and efficient disease detection in plant leaves using Digital Image Processing (DIP) and Deep Learning. It involves several steps:

1. Dataset Collection: The Plant Village Dataset, which includes annotated plant images of Early Blight, Late Blight, and healthy leaves, is obtained as the foundation for training and evaluating the models.

2. Data Cleaning: The dataset undergoes a cleaning process to ensure quality and consistency. This involves removing duplicate or corrupted images, handling missing or erroneous annotations, and ensuring uniform formatting.

3. Data Processing: The dataset is preprocessed using the TensorFlow (tf) dataset framework. The images are normalized, and the data is split into training and testing subsets. Data augmentation techniques like random translations, flips, and rotations are applied to increase training data diversity and robustness.

4. Model Building: A Convolutional Neural Network (CNN) architecture is designed specifically for identifying leaf diseases. The preprocessed dataset is used to train the CNN model, which learns the distinguishing traits and patterns associated with Early Blight, Late Blight, and healthy leaves. State-of-the-art optimization algorithms and loss functions are utilized during training.

5. Conversion to tf Lite Model: The trained CNN model is converted into a TensorFlow Lite (tf Lite) model for easy deployment on mobile devices. This conversion process includes applying quantization techniques to reduce model size and memory usage while maintaining acceptable accuracy levels. Using tf Lite models enables seamless integration into mobile applications.

6. Deployment on Google Cloud: The converted tf Lite model is deployed on the Google Cloud platform, ensuring scalability, accessibility, and efficient processing of disease detection requests. This allows real-time disease diagnosis and provides immediate feedback and actionable insights to farmers.

7. Mobile App Development: A mobile application is developed using React Native and ReactJS frameworks. This app allows farmers to capture leaf images, which are processed using the deployed tf Lite model for disease detection. The app displays diagnosis results, indicating the presence and severity of Early Blight or Late Blight. This information helps farmers make informed decisions for disease management.

By combining DIP, Deep Learning, and cloud-based deployment, this methodology aims to provide an effective and accessible solution for leaf disease detection. It enables farmers to detect diseases promptly, minimize crop losses, and promote sustainable agricultural practices.



## Workflow

This repository provides an overview of the workflow for building a leaf disease detection system using Digital Image Processing (DIP), Deep Learning, and cloud-based deployment. The following sections outline the major components and technologies involved in the process:

### Model Building

In the initial stage, we use TensorFlow, a popular deep learning framework, to design and train a Convolutional Neural Network (CNN) model. This model is specifically designed to identify leaf diseases. The training data is preprocessed and augmented using various techniques provided by the TensorFlow Dataset (tf Dataset) framework.

### Backend Server

To serve the trained model and handle incoming requests, we utilize tf Serving, a TensorFlow serving system that allows efficient model deployment. The backend server is implemented using Fast API, a lightweight Python web framework known for its simplicity and scalability.

### Model Optimization

To facilitate deployment on resource-constrained devices, we employ quantization techniques to optimize the model size and memory usage while maintaining an acceptable level of accuracy. This optimized model is converted into a TensorFlow Lite (tf Lite) model, which is specifically designed for mobile and embedded devices.

### Frontend & Deployment

For the frontend development, we use React JS, a popular JavaScript library for building user interfaces, and React Native, a framework for building native mobile applications. This enables us to develop a user-friendly and intuitive mobile app.

The final step involves deploying the application and the tf Lite model to the Google Cloud Platform (GCP). GCP provides scalability, accessibility, and efficient processing of disease detection requests, ensuring real-time disease diagnosis for farmers.

By following this workflow, we aim to provide an effective and accessible solution for leaf disease detection. The combined use of DIP, Deep Learning, and cloud-based deployment enables farmers to detect and address diseases promptly, minimizing crop losses and promoting sustainable agricultural practices.

For more detailed information and implementation steps, please refer to the documentation and code provided in this repository.
