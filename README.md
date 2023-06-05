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

# Problem
I lost the settings I used for the experiment.  </br>
Results using the code provided by this repository will not match the results of the paper.  </br>
|method|Mean (paper)|Std. Dev (paper)|SSIM (paper)| |Mean (github)|Std. Dev (github)|SSIM (github)| 
|------|---|---|---|---|---|---|---|
|calibnet|1.44|2.87|0.9885| |1.20(-0.24)|2.83(-0.04)|0.9895(+0.0010)|
|lccnet|2.52|5.05|0.9698| |2.37(-0.15)|4.42(-0.63)|0.9728(+0.0030)|
|zhang|0.49|0.89|0.9970| |0.69(+0.41)|1.59(+0.7)|0.9948(-0.0022)|
|ours|0.21|0.71|0.9985| |0.28(+0.07)|1.06(+0.35)|0.9978(-0.0007)|

I have no excuse. Sorry..  </br>
