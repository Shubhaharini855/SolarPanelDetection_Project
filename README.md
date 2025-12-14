Rooftop Solar PV Detection Pipeline:

This project detects rooftop solar photovoltaic panels from high resolution satellite imagery using a two stage deep learning pipeline. The system automatically retrieves satellite images, determines whether solar panels are present, and if present, detects and counts them while producing audit ready outputs.

The pipeline is designed for research and evaluation purposes and follows all open source and dataset usage rules.

Project Overview:

The system works in two stages. First, a convolutional neural network based on ResNet50 determines whether a rooftop contains solar panels. Second, if solar panels are likely present, a YOLOv8 model is used to detect panel locations, count them, and provide explainability artifacts.

The output is a structured JSON file that includes confidence scores, quality control status, and bounding box evidence.

Features:

Automatic satellite image retrieval using Mapbox
Binary classification of solar presence
Object detection and panel counting
Approximate area estimation
Explainability through bounding boxes and audit images
Quality control classification as VERIFIABLE or NOT VERIFIABLE
JSON output for downstream processing

Project Structure Description:

The main pipeline file controls the full workflow from image retrieval to output generation. Trained model weights are stored in the models directory. Downloaded satellite images are stored in the inputs directory. All results including JSON outputs and audit images are saved in the outputs directory.

How to Run the Project:

First create and activate a Python virtual environment.
Install all required dependencies listed in the requirements file.
Create a Mapbox account and obtain a public access token.
Set the Mapbox token as an environment variable on your system.
Ensure the trained ResNet50 and YOLOv8 model files are placed in the models directory.
Run the main pipeline script.
After execution, the outputs directory will contain a results JSON file and audit images.

Input and Output:

Input to the system consists of latitude and longitude coordinates.
For each coordinate, the system retrieves a satellite image and performs inference.

The output is a JSON file containing the sample identifier, location, solar presence flag, confidence score, estimated panel area, quality control status, and metadata about the image source.

Quality Control Logic:

VERIFIABLE is assigned when clear visual evidence of solar panels is detected by the model.
NOT VERIFIABLE is assigned when the image quality is insufficient or no reliable detections are produced due to shadows, low resolution, occlusion, or uncertainty.

Dataset Information and Attribution:

The dataset used to train the YOLO model was obtained from Roboflow Universe.
Dataset name is Custom Workflow Object Detection.
Dataset author is Alfred Weber Institute of Economics.
Dataset license is Creative Commons Attribution 4.0.

The dataset is used in compliance with its license and full rights remain with the original authors. Attribution is provided as required.

Map Data Attribution:

Satellite imagery is provided by Mapbox.
Map data attribution is Mapbox and OpenStreetMap contributors.

License:

The source code of this project is released under the MIT License.
This license grants full rights to use, modify, distribute, and sublicense the software.

The dataset remains under the Creative Commons Attribution 4.0 license and all rights belong to the original dataset creators.

Credits:

Dataset provided by Alfred Weber Institute of Economics via Roboflow Universe.
Deep learning frameworks used include PyTorch and Ultralytics YOLOv8.
Satellite imagery provided by Mapbox.

Caution Warning:

Area estimation is approximate and based on pixel measurements.
Satellite imagery quality may vary due to shadows, occlusions, or outdated captures.
This project is intended for academic, research, and evaluation use.