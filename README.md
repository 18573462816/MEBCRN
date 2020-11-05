# MEBCRN
code for MEBCRN: **Robust water-fat separation based on deep learning model exploring multi-echo nature of mGRE**. For details see:http://dx.doi.org/10.1002/mrm.28586, Magnetic Resonance in Medicine.

## The structure of MEBCRN 
![image](https://github.com/18573462816/MEBCRN/blob/master/MEBCRN.png)



## Separation Sample

![image](https://github.com/18573462816/MEBCRN/blob/master/separation%20sample.png)

## Train and Test

### 1. Environments

- Python (3.6.8)
- PyTorch (0.4.1)
- torchvision (0.2.1)
- NumPy (1.16.2)
- SciPy (1.2.1)
- H5py (2.9.0)
- Scikit_image (0.14.2)
- matplotlib (3.0.3)

### 2. Download the training dataset and the 2012 ISMRM Challenge dataset

Download the training dataset from:
https://data.mendeley.com/datasets/pypbrjnp4j/1

Download the 2012 ISMRM Challenge dataset from:

https://challenge.ismrm.org/node/4

https://challenge.ismrm.org/node/49

Due to the size of the dataset file, the processed trian.mat, trainGroundtruth.mat, val.mat, valGroundtruth.mat, test.mat, testGroundtruth.mat are not provided here. The good news is that these .mat files can be obtained by simply processing the dataset downloaded above using MATLAB. Here, I also provide the dimensions of these .mat files to provide a reference basis for you.

```bash
./data/train.mat: [324, 8, 160, 224], "[num_slice, num_echo, width, height]", complex data
./data/val.mat: [28, 8, 160, 224], "[num_slice, num_echo, width, height]", complex data
./data/test.mat: [32, 8, 160, 224], "[num_slice, num_echo, width, height]", complex data 

./data/trainGroundtruth.mat: [324, 2, 160, 224], "[num_slice, water/fat, width, height]", complex data 
./data/valGroundtruth.mat: [28, 2, 160, 224], "[num_slice, water/fat, width, height]", complex data  
./data/testGroundtruth.mat: [32, 2, 160, 224], "[num_slice, water/fat, width, height]", complex data 
```
### 3. Train and test 

```bash
# train
run train.py
# test
run test.py
```

## Acknowledgements

Acknowledgements will be added later.

