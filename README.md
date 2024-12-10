<div align="center">

<img alt="LOGO" src="https://cdn.jsdelivr.net/gh/fishaudio/fish-diffusion@main/images/logo_512x512.png" width="256" height="256" />

# FADE demo

This is a demo to re-generate the training process of FADE



## 请注意，本项目核心思路来源于FADE识别SRTS模型，本项目近乎为伪代码，需要后续完善

pip install -r requirements.txt  
run python data_generation.py    to generate speech and noises and mix them in different SNR
run train.py   to train the whisper model in different SNR
run test.py    to test the recognition score and get the RRM matrix
run recognition_map.py    to calculate the SRTS  (under hesitation)
run correction.py     to modify the speech recognition function to the average SRT point
