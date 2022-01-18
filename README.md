# Lung Segmentation on Chest X-Ray Images with U-Net
Application of U-Net in Lung Segmentation

# Dataset
Download Dataset from [Chest Xray Masks and Labels Pulmonary Chest X-Ray Defect Detection](https://www.kaggle.com/nikhilpandey360/chest-xray-masks-and-labels)

```
  /data
    /data/Lung Segmentation
      /data/Lung Segmentation/CXR_Png
      /data/Lung Segmentation/masks
```

## Loss Function
Challenge in medical image<br>

- CrossEntropyLoss (Naive Method) <br>
- Dice Loss <br>
  
Different segmentation loss functions implemented in: https://github.com/JunMa11/SegLoss
