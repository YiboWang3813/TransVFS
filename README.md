# The Official Implemention of TransVFS 

**Title:** TransVFS: A Spatio-Temporal Local-Global Transformer for Vision-based Force Sensing during Ultrasound-guided Prostate Biopsy 

**Authors:** Yibo Wang, Zhichao Ye, Mingwei Wen, Huageng Liang, and Xuming Zhang

**Contact:** zxmboshi@hust.edu.cn

--- 

# Environment Requirement 

Our codes mainly rely on these third-party libraries. 

Please ensure they have been installed in your working environment before your using. 
```
python == 3.7.12 
pytorch == 1.7.1 
numpy == 1.19.2 
pandas == 1.1.5 
pillow == 1.8.1 
scipy == 1.5.2 
matplotlib == 3.2.2 
nibabel == 3.2.1 
```

---

# Dataset Download 

You can download our test dataset during the pushing stage on the TRUS dataset of Phantom from [Google Drive](https://drive.google.com/drive/folders/1YC87VCj74Zg5Y9DNk1sKIgx-jvQdtIWb?usp=drive_link). 

When you finish the download, please unpack the test dataset and put it into *dataset* dictionary. 

--- 

# Checkpoint Download 

You can also download our trained model's weight from [Google Drive](https://drive.google.com/drive/folders/1y7FtAf-jz96UTb-Tfbs9ganu0fzdvhkJ?usp=drive_link). 

When you finish the download, please unpack the trained weight and put it into *checkpoint* dictionary. 

--- 

# Evaluate the model 

Please follow this rule to evaluate the trained model. 

```
python main.py --eval \
--dataroot 'your_test_dataset_path' \
--resume 'your_checkpoint_file_path' \
--output_dir 'your_expected_dir_to_save_results' \
--net_name 'TransVFS' \
--depths 'Base' \
--mode 'B' \
--lambda_ 0.05 \
--gap 'Simple' \
--batch_size 8
```

For example, if you follow abormentioned steps to prepare the dataset and the trained weight, you can directly run this command to begin evaluation. The results will be stored in *./output* dictionary. 

```
python main.py --eval \
--dataroot './dataset/test_dataset' \
--resume './checkpoint/checkpoint.pth' \
--output_dir './output' \
--net_name 'TransVFS' \
--depths 'Base' \
--mode 'B' \
--lambda_ 0.05 \
--gap 'Simple' \
--batch_size 8
```

# Notice 

All codes should be used only for academic purpose. 
