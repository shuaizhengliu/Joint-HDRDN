# Joint HDR Denoising and Fusion: A Real-World Mobile HDR Image Dataset (CVPR2023)

## News

- **2022.04.10**: Our Mobile-HDR dataset is uploaded to the Google Drive now.


## Data Download

|              |                        Baidu Netdisk                         |                         Google Drive                        | Description                                                  |
| :----------- | :----------------------------------------------------------: | :----------------------------------------------------------: | ------------------------------------------------------------ |
| Mobile-HDR | todo | [link](https://drive.google.com/drive/folders/1ydUpdeThM2yoZJiCXRB3ZHDVPIykbt2o?usp=share_link) | We offer LDR-HDR data pairs with three-exposure input frames. |


## Mobile-HDR Data structure

```
├── Mobile-HDR
    ├── NPZ_data
         ├── training_npz
              ├── static_translate
              ├── dynamic
         ├── test_npz
              ├── test_withGT
              ├── test_withoutGT

```
( The *.npz data has been processed to be sent network directly. More details will be given before 4.15 )

## Pretrained Model



## Training Phase

### 1. Data prepare

### 2. Training code

## Evaluation Phase

### 1. Data prepare

### 2. Test code

### 3. Visualization by ISP and Tonemapping
## License

This project is licensed under the Apache 2.0. Redistribution and use of the dataset and code for non-commercial purposes should follow this license. 

## Citation

If you find this work useful, please cite:

```
@inproceedings{liu2023mobilehdr,
  title={Joint HDR Denoising and Fusion: A Real-World Mobile HDR Image Dataset},
  author={Liu, Shuaizheng and Zhang, Xindong and Sun, Lingchen and Liang, Zhetong and Zeng, Hui and Zhang, Lei},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2023}
 }
```

## Contact

If you have any question, please feel free to reach me out at `shuaizhengliu21@gmail.com`.
