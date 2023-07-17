# kitti-scc-5dof
Implementation of "A CNN-Based Online Self-Calibration of Binocular Stereo Cameras for Pose Change" </br>
[paper link(Early Access)](https://ieeexplore.ieee.org/document/10138428)

# Weights
Google drive link gives the pre-trained weights of calibnet, lccnet, zhang's network, and our network. </br>
PSMNet's KITTI2012 weights are used for evaluation. </br>
[google drive link](https://drive.google.com/drive/folders/1nnsHzR9cECz9G1VA2lOsapzTY8ksQTI7?usp=drive_link) </br>
[PSMNet link](https://github.com/JiaRenChang/PSMNet)

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

# Run
Check out the readme.md in /scripts

# Visualize Attention
![1](https://github.com/sjg918/kitti-scc-5dof/blob/main/results/000027left.png?raw=true)
![2](https://github.com/sjg918/kitti-scc-5dof/blob/main/results/000027left_.png?raw=true)
![3](https://github.com/sjg918/kitti-scc-5dof/blob/main/results/000027left__.png?raw=true)
</br>

# Problem 1
I lost the settings I used for the experiment.  </br>
Results using the code provided by this repository will not match the results of the paper.  </br>
|method|Mean (paper)|Std. Dev (paper)|SSIM (paper)| |Mean (github)|Std. Dev (github)|SSIM (github)| 
|------|---|---|---|---|---|---|---|
|calibnet|1.44|2.87|0.9885| |1.19(-0.25)|2.89(+0.02)|0.9893(+0.0008)|
|lccnet|2.52|5.05|0.9698| |2.12(-0.40)|4.44(-0.61)|0.9740(+0.0042)|
|zhang|0.49|0.89|0.9970| |0.71(+0.43)|1.72(+0.83)|0.9944(-0.0026)|
|ours|0.21|0.71|0.9985| |0.27(+0.06)|1.00(+0.29)|0.9980(-0.0005)|

# Problem 2
There are personal information in the GY dataset. (Vehicle registration plate, identifiable face) </br>
Therefore, I cannot in any way provide experiments on the GY dataset. </br>

# Special Thanks
[LCCNet: Lidar and Camera Self-Calibration Using Cost Volume Network](https://github.com/LvXudong-HIT/LCCNet)
