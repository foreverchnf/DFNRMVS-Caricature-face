# DFNRMVS-Caricature-face

## 1 Run DFNRMVS in Google Colab
Use run_DFNRMVS_demo.ipynb in Google Colab, you can check the results of DFNRMVS.

## 2 Use DFNRMVS in local Ubuntu 18.04 computer
### 2.1 Run official demo
(1) Follow the guide in [DFNRMVS](https://github.com/zqbai-jeremy/DFNRMVS) to compile environment.
#### Prerequisite
- Nvidia GPU resource available and Nvidia drive correctly installed 

(2) Run demo.py successfully in your machine.

### 2.2 Make Caricature face
#### Prerequisite
- python packages
  * dlib
- Be sure 
  * Run it on the same path as the official demo.py of [DFNRMVS](https://github.com/zqbai-jeremy/DFNRMVS)
  * [shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) on path
  * DFNRMVS_Caricature_utils.py on path
  
#### Run DFNRMVS_Caricature.py to make a Caricature face based on DFNRMVS reconstruction

## Results
<img src="https://github.com/foreverchnf/DFNRMVS-Caricature-face/blob/master/face_1.jpg" width="300" height="300" alt="face"/>
<img src="https://github.com/foreverchnf/DFNRMVS-Caricature-face/blob/master/2020-11-06%2013-28-27%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png" width="300" height="300" alt="face"/>
<img src="https://github.com/foreverchnf/DFNRMVS-Caricature-face/blob/master/defor_face.png" width="300" height="300" alt="face"/>
<img src="https://github.com/foreverchnf/DFNRMVS-Caricature-face/blob/master/Caricature.jpg" width="300" height="300" alt="face"/>

 

  

