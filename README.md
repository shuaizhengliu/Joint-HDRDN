# Joint HDR Denoising and Fusion: A Real-World Mobile HDR Image Dataset (CVPR2023)

[Paper](https://drive.google.com/file/d/1EnFFwjnHGfKliRTnAMRGZIX-yBN8isx_/view?usp=sharing) | [Supp](https://drive.google.com/file/d/17zemSVqbpmoe5sqqxgjaYEV2IA7x-YAi/view?usp=sharing)
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

## Explanation for each data file

For each .npz file in above folders, it contains Raw data of three-exposures LDRs and corresponding HDR GT in raw format (except for the test data in test_withoutGT folder). The Raw data of each exposure has been processed through black level correction and normalization. We split the RAW data with shape of H and W into four sub-matrix with shape of H/2 and W/2 based on the bayer pattern. The four sub-matrix are concatenated as four-channel matrix in B-G-G-R sequential. Furthermore, we obtain the corresponding luminance-aligned data for each LDR by dividing it with exposuse ratio, which also owns four channel finally. Lastly, we concatenate the origianl raw data and the corresponding luminance-aligned data as an eight-channel matrix for each LDR. Since the HDR GT does not need the luminace alignment, its has only four channels. Specifically, the components are arranged as below.

(The specific read operation for training has been offered in dataset.py)

```
#  Key_name Column_index1              Column_index2
---['sht']  [0:4] (original raw data)  [4:8] (luminace-aligned data)  # The data of short exposure
---['mid']  [0:4] (original raw data)  [4:8] (luminace-aligned data)  # The data of mid exposure
---['lng']  [0:4] (original raw data)  [4:8] (luminace-aligned data)  # The data of long exposure
---['hdr']  [0:4] (hdr GT)

```

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
