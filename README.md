# DFNRMVS-Caricature-face

## 1 Run DFNRMVS in Google Colab
Use run_DFNRMVS_demo.ipynb in Google Colab, you can check the results of DFNRMVS.

## 2 Use DFNRMVS in Ubuntu 18.04 with Nvidia GPU
### 2.1 Run official demo
Follow the guide in [DFNRMVS](https://github.com/zqbai-jeremy/DFNRMVS) to compile environment.
#### Prerequisite
- Nvidia GPU resource available and Nvidia drive correctly installed 

Run demo.py successfully in your machine.

### 2.2 Make Caricature face
#### Prerequisite
- python packages
  * dlib
- Be sure 
  * Run it on the same path as the official demo.py of [DFNRMVS](https://github.com/zqbai-jeremy/DFNRMVS)
  * [shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) on path
  * DFNRMVS_caricature_utils.py on path
  
Run DFNRMVS_caricature.py to make a Caricature face based on DFNRMVS reconstruction

 

  

