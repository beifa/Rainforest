# Rainforest Connection Species Audio Detection [link](https://www.kaggle.com/c/rfcx-species-audio-detection/overview)

## 70th place solution

Thanks to Kaggle and hosts for this very interesting competition with a tricky setup. My solution is an ensemble of several CNNs, which take a mel spectrogram.

### DATA
I made 4 dataset which gave the best scored 
 - preprocess/make-data-img-sr48power2mel260-reinforest.ipynb
 - preprocess/make-data-img-sr32power2mel384-ch-f0-9-f1-1-power-to-db.ipynb

 little change timefrom 6 sec to 9 sec

 -  preprocess/exp-make-data-img-sr48power2mel260-reinforest.ipynb
 - preprocess/exp-make-data-img-sr32power2mel384-ff111-pow.ipynb

To reproduce the result, you need to use an ensemble of 6 of my models 4 made with kernels 2 no (since I used a pytorch(AUTOMATIC MIXED PRECISION ) the result may have become a little worse)

### Predict
```
s_880
 - sub2_BLite4_sr32power2mel384_notchageSize_folddata_addlayers_RAND_best_None
  train.py need change to:
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        mode='min',
                                                        factor=0.9, 
                                                        patience=3,
                                                        verbose=True)  
    and use sr48power2mel260 data

s_878
- notebook/train_default_260_878&_addlayers_852.ipynb
s_882
- notebook/best-rand-exper-380-888-882.ipynb
s_882_seed887
- notebook/best-rand-exper-380-888-882.ipynb

train.py use rand and exp data
  s_871 
  - sub2_BLite4_sr48power2mel260_RAND_schedulerGWS_CosineALR_best_NoneEXPDATA_871.csv')
  s_873 
  - sub2_BLite4_sr32power2mel384_RAND_schedulerGWS_CosineALR_best_NoneEXPDATA_873.csv')   
 
 Best score:
  70 place
  private : 0.91191
  public  : 0.90821

  submission = (s_880.iloc[:,1:] + 
                s_878.iloc[:,1:] + 
                s_882.iloc[:,1:] + 
                s_882_seed887.iloc[:,1:]/5)  +               
                s_871_.iloc[:, 1:] +
                s_873.iloc[:, 1:]) / 6
OR

best my mean(s-880s-878s-882s-887)
82 place
private : 0.90718
public  : 0.90401                   
                   
```
### Model, > 300 submit
I tried many models, the best in my case turned out to be tf_efficientnet_lite4
In the EXP dataset, I used a 9 sec crop (default 6 sec) then when I trained the model I randomly in 9 sec take 6 sec.
I use true_positive label.

Tune model by optuna:
  - head layers
  - argumentation
  - scheduler, optimizer and other

Attempts to use argumentation and TTA did not give an increase score

Structure:
 - learn curve img
 - no_magic -- my experiments
 - notebook - kernel from kaggle
    - learn_curve.ipynb (visual)
    - und-data-rainforest.ipynb (Understand competition)
 - preprocess -- make data
 - src -- model

### Visual how change lerning curve

Start, Sat_Jan_2_12:34:35_2021
<img src= "./learn curve img/Sat_Jan_2_12:34:35_2021_log_LiteB4_576x224_30_kfold_Plateau_Adam_pos_weights_lr_default_pretrain_TP.png">
Fri_Feb_5_23:16:05_2021
<img src= "./learn curve img/Fri_Feb_5_23:16:05_2021_log_BLite4_sr32power2mel384_notchageSize_folddata_addlayers_RAND.png">
End, Tue_Feb_16_14:22:27_2021
<img src= "./learn curve img/Tue_Feb_16_14:22:27_2021_log_BLite4_sr48power2mel260_RAND_schedulerGWS_CosineALR.png">