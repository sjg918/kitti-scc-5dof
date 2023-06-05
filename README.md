# kitti-scc-5dof
Implementation of "A CNN-Based Online Self-Calibration of Binocular Stereo Cameras for Pose Change"

# Weights
This link gives the weights of calibnet, lccnet, zhang's network, and our network. </br>
[google drive link](https://drive.google.com/drive/folders/1nnsHzR9cECz9G1VA2lOsapzTY8ksQTI7?usp=drive_link)

# Setup
Download the kitti odometry dataset. </br>
Replace all calib.txt in data folder with calib.txt downloaded from "Download odometry data set (calibration files, 1 MB)". </br>
(('Tr') line does not exist, giving an error.) </br>

Open the cfg file in the CFG folder and modify the paths. </br>

# Requirement
python >= 3.10.9. I worked in an anaconda environment. </br>
pytorch >= 1.13.1 (https://pytorch.org/) </br>
opencv (pip install opencv-python) </br>
easydict (pip install easydict) </br>
numba (pip install numba) </br>
matplotlib (pip install matplotlib) </br>

GPU with more than 20 GB of memory </br>

# Run
Check out the readme.md in /scripts
