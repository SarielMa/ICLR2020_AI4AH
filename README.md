# IMPROVE ROBUSTNESS OF DNN FOR ECG SIGNAL CLASSIFICATION:A NOISE-TO-SIGNAL RATIO PERSPECTIVE
This work was conducted at Department of Computer Science, University of Miami and has been accepted as by ICLR2020 workshop AI for affordable health https://sites.google.com/view/ai4ah-iclr2020/schedule. Paper is available at https://arxiv.org/abs/2005.09134. If you find our codes useful, we kindly ask you to cite our work.

## Abstract
Electrocardiogram (ECG) is the most widely used diagnostic tool to monitor the
condition of the cardiovascular system. Deep neural networks (DNNs), have been
developed in many research labs for automatic interpretation of ECG signals to
identify potential abnormalities in patient hearts. Studies have shown that given
a sufficiently large amount of data, the classification accuracy of DNNs could
reach human-expert cardiologist level. A DNN-based automated ECG diagnostic system would be an affordable solution for patients in developing countries
where human-expert cardiologist are lacking. However, despite of the excellent
performance in classification accuracy, it has been shown that DNNs are highly
vulnerable to adversarial attacks: subtle changes in input of a DNN can lead to
a wrong classification output with high confidence. Thus, it is challenging and
essential to improve adversarial robustness of DNNs for ECG signal classification
a life-critical application. In this work, we proposed to improve DNN robustness
from the perspective of noise-to-signal ratio (NSR) and developed two methods
to minimize NSR during training process. We evaluated the proposed methods on
PhysionNets MIT-BIH dataset, and the results show that our proposed methods
lead to an enhancement in robustness against PGD adversarial attack and SPSA
attack, with a minimal change in accuracy on clean data.

## Keywords: ECG, DNN, robustness, adversarial noises

# Environment
Python version==3.7; Pytorch version==1.2; Operation system: Win 10 or CentOS 7.

# Prepare dataset
Download data from https://www.kaggle.com/shayanfazeli/heartbeat. Put the csv files at /ecg/ and run preprocess.py.

# Questions
If you have any question, please contact the authors (l.ma@miami.edu or liang.liang@miami.edu).
