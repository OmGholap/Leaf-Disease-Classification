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


## Workflow

This repository presents a detailed workflow for building a potato leaf disease detection system using Digital Image Processing (DIP) and Deep Learning. The workflow consists of several steps:

### Data Collection and Preparation

1. Collect a diverse dataset of digital images that includes healthy potato leaves, as well as samples of leaves affected by Early Blight and Late Blight.
2. Annotate the dataset by labeling each image with the corresponding disease type (healthy, Early Blight, or Late Blight).
3. Create training, validation, and testing sets from the annotated dataset.

### Model Building

1. Construct a Convolutional Neural Network (CNN) model using TensorFlow, a popular deep learning framework.
2. Design the CNN model with suitable layers such as convolutional, pooling, and fully connected layers.
3. Train the model using the training dataset to learn the features that distinguish healthy leaves from diseased ones.
4. Optimize the model's parameters to improve its performance.
5. Enhance the model's resilience and generalizability by applying data augmentation techniques, including random rotations, flips, and zooms.
6. Use the TensorFlow Dataset (tf.data) API to efficiently load, preprocess, and batch the dataset.

### Model Evaluation and Optimization

1. Evaluate the performance of the trained model using the validation dataset, considering metrics such as accuracy, precision, recall, and F1 score.
2. Adjust the model's hyperparameters and architecture based on the evaluation results.
3. Apply model optimization techniques, such as quantization, to reduce the model's memory footprint and accelerate inference.
4. Convert the optimized model to TensorFlow Lite format for deployment on devices with limited resources.

### Backend Server

1. Implement an API endpoint that accepts potato leaf images as input and performs inference using the trained model.
2. Utilize TensorFlow Serving, a system for serving TensorFlow models, to enable efficient model deployment and inference.

### Frontend Development

1. Design an intuitive user interface that allows users to upload potato leaf images and displays the disease detection results.
2. Implement image preprocessing on the frontend, including resizing and normalization, before sending the images to the backend server for inference.
3. Develop a responsive and user-friendly interface that provides visual feedback on disease detection results and confidence scores.

### Deployment to Google Cloud Platform (GCP)

1. Deploy the backend server, built with FastAPI and TensorFlow Serving, to a cloud-based environment on the Google Cloud Platform (GCP).
2. Set up the necessary infrastructure, such as a virtual machine instance or Kubernetes cluster, to host the backend server.
3. Deploy the frontend application to GCP and configure it to interact with the backend server API.
4. Ensure appropriate security measures, such as authentication and encryption, to protect sensitive data during communication between the frontend and backend.

By following this workflow, you can build an effective and user-friendly potato leaf disease detection system. For detailed information and step-by-step instructions, refer to the documentation and code provided in this repository.

## Tools and Technologies Used

This repository utilizes a range of tools and technologies to build a potato disease detection system. Here is a brief overview:

### Visual Studio Code (VS Code)

Visual Studio Code is the chosen code editor for writing, editing, and organizing the codebase. It provides a rich ecosystem of extensions from the marketplace, enhancing productivity and supporting various programming languages and frameworks.

### TensorFlow

TensorFlow, an open-source deep learning framework, is employed for building and training Convolutional Neural Network (CNN) models for potato disease detection. It offers a wide range of pre-built functions and tools to simplify model development, optimization, and evaluation. TensorFlow's high-level APIs, such as Keras, make it easier to design and train deep learning models.

### Python

Python is the primary programming language used to implement the potato disease detection system. Python's simplicity and readability make it well-suited for tasks such as image processing, data manipulation, and model training. The extensive ecosystem of Python libraries and packages, including NumPy, matplotlib, and scikit-learn, is leveraged for image loading, data preprocessing, visualization, and evaluation.

### CNN (Convolutional Neural Network)

Convolutional Neural Networks (CNNs) are utilized for detecting potato diseases. CNNs are specifically designed for image processing tasks and can automatically learn and extract relevant features from input images. Their ability to capture spatial dependencies and hierarchical representations makes them suitable for detecting complex patterns and structures in potato leaf images. Techniques like transfer learning, which utilize pre-trained CNN models, are employed to accelerate model development and improve performance with limited training data.

### NumPy

NumPy, a fundamental Python library for numerical computing, is used to convert potato leaf images into arrays. The multidimensional array objects provided by NumPy are leveraged to represent and manipulate the pixel values of the images efficiently.

### React.js and React Native

React.js is employed for building the frontend application, providing an efficient and responsive user interface for the potato disease detection system. React's virtual DOM concept enables efficient updating and rendering of UI elements, resulting in a smooth user experience. React's state management capabilities handle user interactions, manage image uploads, and display disease detection results. React Native, a framework for building native mobile applications, is utilized to develop a mobile version of the application, providing a native-like experience for mobile users.

### FastAPI

FastAPI, a high-performance web framework, is used as the backend for creating Python APIs. It enables the development of efficient and scalable API endpoints to handle image inputs and perform inference using the trained models.

By leveraging these tools and technologies, the potato disease detection system is built with efficiency, scalability, and user experience in mind. For more detailed information and implementation instructions, please refer to the documentation and code provided in this repository.

## Conclusions and Future Scope

The research and development of the potato disease detection system using deep learning and digital image processing have yielded promising results. We have successfully created an automated system capable of accurately diagnosing and classifying Early Blight and Late Blight diseases in potato plants, leveraging powerful tools such as TensorFlow, CNNs, and data augmentation. This system empowers farmers with an efficient and reliable tool for timely disease detection, minimizing crop losses, and ensuring food security.

However, there are areas where further improvements can be made. Expanding the dataset to include a wider range of potato varieties and disease severities would enhance the accuracy and generalizability of the system. Exploring advanced deep learning techniques, such as ensemble models and attention mechanisms, can help capture more intricate patterns and improve disease identification performance.

Looking towards the future, there are several avenues for further enhancement. Integrating cloud-based solutions like TensorFlow Serving can enable scalable and efficient deployment of the disease detection system. Incorporating Internet of Things (IoT) and remote sensing technologies for continuous crop monitoring can provide real-time insights and enable proactive disease management. Additionally, considering sustainability, integrating sensor technologies and exploring hybrid mobile frameworks can deliver a seamless user experience while minimizing environmental impact.

By addressing these shortcomings and pursuing the outlined scope for improvement, we can advance the field of potato disease detection, foster sustainable agricultural practices, and provide valuable support to farmers in mitigating economic losses worldwide.

For more detailed information on the system's implementation, results, and future directions, please refer to the documentation and code provided in this repository.
