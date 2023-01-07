# AutoEncoder Model: Anomaly Detection

<p align='justify'>
Anomaly Detection is one the most interesting subjects in machine learning, and it uses in various areas, such as industries, healthcare, and many other fields. Many articles implement different models to detect anomalies in an image with the best accuracy. In this project, we use an AutoEncoder model to recognize abnormalities in a picture. An AutoEncoder model is a deep model that consists of two different stages. The first stage is an encoder to encode input from high dimensions to low dimensions, and then the second stage, called a decoder, reconstructs the input from low dimensions. The structure of an AutoEncoder is shown in the below picture.
</p>

<br />
<p align="center">
<img src="https://github.com/HosseinPAI/AutoEncoder-Anomaly-Detection-/blob/master/.idea/pics/model.png" alt="AutoEncoder Model"  width="800" height='300'/>
</p>

<p align='justify'>
When we train an AutoEncoder model with correct images, we teach the model how to reconstruct images and learn important features from correct pictures without any anomalies. Then, suppose we impose an image with an anomaly on the trained model. In that case, it can reconstruct the actual image, so the differences between the anomaly input image and the reconstructed output show us anomalies. 
</p>

<p align='justify'>
In this project, we implement an autoencoder with PyTorch to detect anomalies. <a href="https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937545-1629951845/hazelnut.tar.xz">The Hazelnut dataset</a> has been used to train this model. In this dataset, there are three various images. One of them is good images without any anomalies to train the model. The other one is anomaly images to test the model, and the last part is related to ground truth images to detect and compare anomalies. These three images are shown below.
</p>

<br />
<p align="center">
<img src="https://github.com/HosseinPAI/AutoEncoder-Anomaly-Detection-/blob/master/.idea/pics/good.png" alt="Good Image" width="200" height='200'/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/HosseinPAI/AutoEncoder-Anomaly-Detection-/blob/master/.idea/pics/Anomaly.png" alt="Anomaly Image" width="200" height='200'/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/HosseinPAI/AutoEncoder-Anomaly-Detection-/blob/master/.idea/pics/Ground_truth.png" alt="Ground Truth" width="200" height='200'/> </p>
<p align="center">
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Good Image &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Anomaly Image &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Ground Truth Image</p>

<br />
<p align="justify">
You can run the "main.py" file to train and see results. In this project, we have different modes to learn the effects of Batchnormalization and input image size on the model. You can choose which mode you want to run when you run the main file. Before you run the model, you need to set the dataset direction in the main file. The sample result of this model is shown in the below picture.
</p>

<br />
<p align="center">
<img src="https://github.com/HosseinPAI/AutoEncoder-Anomaly-Detection-/blob/master/.idea/pics/result.png" alt="Result"/>
</p>

<br />
All necessary libraries are put in the 'requirement.txt' file.    
