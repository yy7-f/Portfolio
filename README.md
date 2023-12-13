# Machine Learning and Computer Vision Projects

## Neural Networks and Machine Learning for Modeling of Electrical Parameters and Losses in Electric Vehicle

Artificial neural network and other machine learning models including k-nearest neighbors, decision tree, random forest, and multiple linear regression with a quadratic model are developed to predict electrical parameters and losses as new prediction approaches for the performance of Volvo Cars’ electric vehicles and evaluate their performance.  
Grid search with 5-fold cross validation was implemented to optimize hyperparameters of artificial neuralnetwork and machine learning models.   
The artificial neural network models performed the best in MSE and R-squared scores for all the electrical parameters and loss prediction. The results indicate that artificial neural networks are more successful at handling complicated nonlinear relationships like those seen in electrical systems compared with other machine learning algorithms.  
Also, PCA analysis and correlation matrix analysis are conducted.  
This projected was implemented at Volvo Cars Corporation as a Master thesis project.  
[Github Link](https://github.com/yy7-f/ML_for_EV_parameters_and_Losses)  
[Publication Link](https://www.diva-portal.org/smash/record.jsf?pid=diva2%3A1789150&dswid=1220)


![EV](/images/electricvehicle.jpg)

## Brain Tumor Segmentation using Unet
Unet models optimized with the Dice Loss and binary cross entropy loss are developed for brain tumor segmentation using brain MRI images dataset. The dataset contains brain MR images together with manual FLAIR abnormality segmentation masks. After 25 epochs training, the dice coefficient acheived around 0.8.  
[Github Link](https://github.com/yy7-f/Unet-Brain-Segmentation)

![Brain](/images/brainmri.png)

## Maintenance Prediction for Turbofan Jet Engine
LSTM, RNN, and 1D-CNN models were developed for regression and classification tasks using time series data. Dimension reduction was also conducted through data analysis. Regression objective is Predicting the remaining useful life (RUL) of a machine. Classification objective is Predicting the failure of machine in upcoming specific periods.  
[Github Link](https://github.com/yy7-f/Maintenance-Prediction-for-Turbofan-Jet-Engine)

![image](https://github.com/yy7-f/Portfolio/assets/76237852/193d9cb9-726b-48c8-9e9d-2b599ac4a481)


## Virtual sorting system using image classfication, vision algorithm, virtual modeling, PLC programming, and robot operation
Product sorting system in virtual production line is developed in this project.
The sorting production line is built using virtual modeling (Simumatik), PLC programming (Codesys), vision algorithm (Python, OpenCV, OPCUA), image classification (CNN, Transfer learning using VGG16, and HoG + ML), and robot operation (ABB Robot studio).  
[Github Link](https://github.com/yy7-f/CNN_TransferLearning_HoG_ML_for_Virtual_Sorting_System)

![Virtual](/images/virtual_sorting.png)

## DCGAN for generating new data using MNIST dataset

Developed DCGANs (Deep Convolutional Generative Adversarial Networks) with Pytorch to generate new data using MNIST handwritten digit data.  
Architecture of DCGANs  
• Replace all pooling layers with strided convolutions  
• Use batchnorm in both the generator and the discriminator  
• Remove fully connected hidden layers  
• Use ReLU activation in generator for all layers except for the output (Tanh is used for the output)  
• Use LeakyReLU activation in the discriminator for all layers  
  
[Github Link](https://github.com/yy7-f/DCGAN-MNIST-handwritten-digit)

<img width="750" alt="image" src="https://github.com/yy7-f/Portfolio/assets/76237852/5b769b81-5438-4f0a-92ef-bb36974ce17e">


## Hand Tracking and Finger Count using OpenCV 


e number of fingers.
- Hand Tracking:  
Detect and track 21 3D hand landmarks using OpenCV and mediapipe liblaries.
- Finger Counter:  
Count the number of fingers Using hand tracking module. This is useful if you want the computer to perform tasks based on the number of fingers.

[Github Link](https://github.com/yy7-f/Computer-Vision-Hand-Tracking-and-Finger-Count)

<img width="667" alt="image" src="https://github.com/yy7-f/Computer-Vision-Hand-Tracking-and-Finger-Count/assets/76237852/3dbb8459-73c3-44f5-b652-efdf5aea6c55">

## Car Rating Classification
Developed machine learning algorithms to predict the class value of a particular car based on the vehicle’s features. Categorical data was encoded followed by order. Three algorithms including k-nearest neighbor, decision tree, and random forest are executed with 10-fold cross-validation. The results are visualized and compared using confusion matrices and box plots.  
[Github Link](https://github.com/yy7-f/Classification_Car_evaluation_UCI_repository)


![image](https://github.com/yy7-f/Portfolio/assets/76237852/a89538f8-edc8-4055-b4f9-8560ba7ec24c)

## Stable diffusion for text-to-image using Hugging face
Generated images from text using a pre-trained stable diffusion model (Stable Diffusion v2) on Hugging face.  
  
Stable Diffusion is composed of three major components:
- U-Net  
- VAE  
- Text Encoder (Transformer)

Stable Diffusion can efficiently generate high-resolution images by training a diffusion model on the VAE latent space.
Text Encoder is trained on CLIP.
Text conditioning is performed by Cross-Attention in U-Net.  
[Github Link](https://github.com/yy7-f/Stable-Diffusion-text-to-image)

![image](https://github.com/yy7-f/Portfolio/assets/76237852/69d93160-d2d9-45f2-b6e0-8b39f6fb19f4)



# Certificate
- AI Engineering Professional Certificate (IBM)
- Practical Data Science on the AWS Cloud Specialization Certificate (AWS, DeepLearning.AI)
- Generative Adversarial Networks Specialization Certificate (DeepLearning.AI)
- Analyze Datasets and Train ML Models using AutoML (AWS, DeepLearning.AI)
- Build, Train, and Deploy ML Pipelines using BERT (AWS, DeepLearning.AI)
- Optimize ML Models and Deploy Human-in-the-Loop Pipelines (AWS, DeepLearning.AI)
- ROS2 Robotics Developer Course - Using ROS2 In Python (Udemy)
- Python 3: Deep Dive (Object Oriented Programming) (Udemy)
- Apply Generative Adversarial Networks (GANs) (DeepLearning.AI)
- Advanced Learning Algorithms (DeepLearning.AI)
- Supervised Machine Learning: Regression and Classification (DeepLearning.AI)
- Deep Neural Networks with PyTorch (IBM)
- Building Deep Learning Models with TensorFlow (IBM)
- Production Machine Learning Systems on Google Cloud (Google)
- Data Analysis with R Programming (Google)
- Share Data Through the Art of Visualization (Google)
