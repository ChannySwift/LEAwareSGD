# Introduction
This is the code for the ICCV 2025 Paper: Adversarial Data Augmentation for Single Domain Generalization via Lyapunov Exponent-Guided Optimization
# Data
## Datasets:PACS
Download the PACS dataset (h5py files pre-read) from https://drive.google.com/drive/folders/0B6x7gtvErXgfUU1WcGY5SzdwZVk?resourcekey=0-2fvpQY_QSyJf2uIECzqPuQ, 

rename the folder as PACS.

Use the path of the PACS folder as the value of PACS_DATA_FOLDER in config.py.

## Datasets: OfficeHome
Download https://www.kaggle.com/datasets/iamjanvijay/officehome  

Split OfficeHome: https://www.alipan.com/s/hXQYxcwp2Vx

Use the path of the OfficeHome folder as the value of OfficeHome_DATA_FOLDER in config.py.

## Datasets: VLCS
Download https://www.kaggle.com/datasets/iamjanvijay/vlcsdataset  

Split VLCS: https://www.alipan.com/s/bc4eZk494xM

Use the path of the VLCS folder as the value of VLCS_DATA_FOLDER in config.py.

## Datasets: TerraIncognita
Download https://www.alipan.com/s/kLC6rrrxWTJ

location_38 can't share. From https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_all_images_sm.tar.gz to get location_38.

## Datasets: DomainNet
Download (cleaned version) https://ai.bu.edu/M3SDA/

Use the path of the DomainNet folder as the value of DomainNet_DATA_FOLDER in config.py.

## Datasets: Fundus
We follow GDRBench https://github.com/chehx/DGDR/blob/main/GDRBench/README.md. 

https://pan.baidu.com/s/1kprxasgXLT36uTzlhBZkKw   Password: uae9

MESSIDOR and FGADR 

Due to the license, these databases can not be further distributed.

MESSIDOR: https://www.adcis.net/en/third-party/messidor2/

FGADR: https://csyizhou.github.io/FGADR/

# Experiments
```bash run_command.sh```

# Citation
Please consider citing this paper if you find the code helpful.

```
@inproceedings{zhang2025adversarial,
  title={Adversarial Data Augmentation for Single Domain Generalization via Lyapunov Exponent-Guided Optimization},
  author={Zhang, Zuyu and Chen, Ning and Liu, Yongshan and Zhang, Qinghua and Zhang, Xu},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={552--561},
  year={2025}
}
```
