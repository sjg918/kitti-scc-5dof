# trainkitti.py
Training takes one day. </br>
Miscalibration is done using gpu. If your gpu memory is low, you can run it with cpu. . . </br>

# val_kitti.py
Evaluation with iterative refinement and temporal filtering applied. </br>
I used the same noise for all experiments to control the experiments. (check /gendata/100miscalib~.txt) </br>
To evaluate the different methods you need to modify lines 14-25, 449-465 of the code. </br>
Additionally, this code can provide visual results. </br>
You can get visualization results by input the path argument to the mkPSMdispmap function. </br>

# vis_kitti.py
Modify lines 200 of the code.. </br>
Angle values must be real numbers between -2.5 and 2.5. </br>
