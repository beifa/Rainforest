default parameters for image:
  - f_min, f_max = df.f_min.min() * 0.8, df.f_max.max() * 1.2
  - SAMPLERATE = 48000
  - add = 3 * sr   6 sec frame
  - mel_spec = librosa.feature.melspectrogram(cut_data,
                                                n_fft=2048,
                                                hop_length=512,
                                                sr=sr,
                                                fmin=f_min,
                                                fmax=f_max,
                                                power=1.5,
                                                n_mels = 224
                                              )
  - mel_spec = librosa.power_to_db(mel_spec).astype(np.float32)
  - test dim (10, 224, 563) 
==============================================Tune
1. (samplerate = 32000
   image(224, 376)
   all default)
    
   s1_376x224 lite4 sr 32000.csv --> 0.812
   s2_376x224 lite4 sr 32000.csv --> 0.809
2. (samplerate = 22050
   image(224, 259) resise 260*260
   all default)
   s1_sr22050_224x224.csv --> 0.775

3. samplerate 32000, mels = 384, power =2
  time 3, image = (384, 376)== 384*384
  s2_sr32power2mel384.csv --> 0.818

4. time 5(reinforest_img_spectrogram_v1_mel224)
  s1_(224 626) 10 sec blite4.csv --> 0.803
  s2_(224 626) 10 sec blite4.csv --> 0.807

5. sr32time4alldefault
  s2_sr32time4alldefault blite4 224501.csv --> 0.800

6. sr32time2alldefault
  s2_sr32time2alldefault blite4 224251.csv -->0.744

7. sr32time20alldefault
  s2_sr32time20alldefault.csv --> 0.821

8. sr32time14alldefault
  s2_sr32time14alldefault.csv --> 0.794

9. sr48power2mel384
  s2_48power2mel384.csv --> 0.718 black lines in image

10. sr32time20alldefault_v1 power 1
  s2_sr32time20alldefault_power1.csv --> 0.816

11. sr32time20alldefault power 2
  s2_sr32time20alldefault_power2.csv --> 0.798


12. sr48power1mel384f ft4096*2 hope 512
  s2_datarain-sr48power2mel384_v2.csv --> 0.792

13. power=1.35,
  s2_sr32time20alldefault_power1.35.csv --> 0.798

14. power=1.15
  s2_sr32time20alldefault_power1.15.csv --> 0.816

15. sr48time20alldefault_power1
s2_sr48time20alldefault-power1.csv -->0.793

16. sr32time20alldefault_power3
s2_sr32time20alldefault-power3.csv -->0.797


я криворукий надо пертестить * 2 дулировал картинку

17. 
s2_sr32time10chage_timemake_pow1.csv --> 0.771

18. sr32time20chage_timemake_pow1 --> 0.542

19. peretest_change_time_sr32time20alldefault
  s2_peretest-change-time-sr32time20alldefault.csv --> 0.783

20. peretest_best_sr32time20alldefault
  s2_peretest-best-sr32time20alldefault.csv --> 0.829

21. corrects?
s2_peretest-best-sr32time20alldefault_CORRECT.csv --> 0.822

22. peretes-sr32power2mel384-reinforest
s2_datarainsr48power2mel384.csv --> 0.795

s2_2mel384.csv --> 0.795


23. sr32power2mel384
s2_corrects.csv --> 0.796

---------------change data

24. make_img_sr32power2mel384_reinforest
  s1.csv --> 0.822
  s2.csv- change data chage folds, 384*387, tp --> 0.844

25. sr32power1.5mel128time20_TP
s2_r32power1-5mel128time20-tp.csv --> 0.816

26. sr32power2mel384_notuse_powertodb
s2_sr32power2mel384-notuse-power.csv --> 0.859

27. sr48power2mel260 !!! powertodb !!!
s2_sr48power2mel260.csv --> 0.874

28. sr32power1.5mel128time10_TP
s2_sr32power1-5mel128time10-tp.csv --> 0.826

29. sr32power1.5mel384
s2_sr32power1-5mel384-notuse-powerto.csv --> 0.842

30. sr32power3mel384
s2_sr32power3mel384-notuse-pow.csv --> 0.846

31. sr32power1mel384
s2_sr32power1mel384-notuse-powe.csv --> 0.825

32. make_img_sr48power2mel260_notuse_powertodb
s2_sr48power2mel260-notuse-pow.csv --> 0.841

33. sr32power2.5mel384_notuse_powertodb
s2_sr32power2-5mel384-notuse-pow.csv --> 0.854

33. make_img_sr32power1.5mel128time6_TP_reinforest
s2_sr32power1-5mel128time6-tp.csv --> 0.806

34.
s2_sr32power2mel384_power_to_db_ch time 4.csv -->0.806

35.
s2_sr32power2mel384_ch_f0.9_f1.1.csv --> 0.870

36. make_img_sr32power2mel384_ch_f0.6_f1.4_power_to_db  
s2_sr32power2mel384_ch_f0.6_f1.4_power_to_db.csv --> 0.817

37. make_img_sr32power2mel384_ch_f1.1_f0.9_pow 
s2_sr32power2mel384_ch_f1.1_f0.9.csv --> 0.845

38 sr32power2mel384_ch_f1_f1_pow 
s2_sr32power2mel384-ch-f1-f1-pow.csv --> 0.855

39 sr32power2mel384_ch_f0.95_f1.05
s2_sr32power2mel384-ch-f0-95-f1-05-pow.csv --> 0.857

40. ( s2_sr48power2mel260.csv --> 0.874 )
ch_f0.9_f1.1.
s2_sr48power2mel260-ch-f0-9-f1-1.csv --> 0.849


40. ( s2_sr48power2mel260.csv --> 0.874 ) peretest
s2_sr48power2mel260.csv --> 0.874

41 sr32power2mel384-ch-f0-9-f1-1-pow FULL
2_sr32power2mel384-ch-f0-9-f1-1-pow full.csv --> 0.328

42 s2_sr48power2mel260 Full
s2_full-img-sr48power2mel260.csv --> 0.290




commit
sub2_BLite4_sr32power2mel384_notchageSize_best_None.csv --> 0.859


1. add sample not add fp( tp * 2)


2. add duplicate
FULL_img_sr32power2mel384, add to tp all duplicate from fp, fold size = (1716, 428) --> 0.833

3. full but drop duplicate fp

after
1. make pred one fold
s2_sr32power2mel384-ch-f0-9-f1-1-pow_FOLD_0.csv --> 0.743

s2_full-img-sr48power2mel260_FOLD_0.csv --> 0.806



s2_sr48power2mel260-p2 x2.csv --> 0.843, ***ADD TP * 2

sub2_BLite4_sr32power2mel384_notchageSize_best_None.csv --> 0.860

s2_sr32power2mel384-ch-f0-9-f1-1-pow.csv --> 0.840

s2_duplicate.csv --> 0.873


sub2_BLite4_sr48power2mel260_notchageSize_best_None.csv --> 0.841
sub2_BLite4_sr48power2mel260_notchageSize_best_None.csv --> 0.690 not correct data retest


IDEA folds

s2_sr32power2mel384ch_f0.9_f1.1 bin fold 15.csv --> 0.856

s2_sr48power2mel260 bin fold 16.csv --> 0.850

s2_add_fp_mark_1.csv --> 0.416

s2.csv  tp + fp in tp and add duplicate tp by add fp --> 0.827

s2_sr48power2mel260 bin fold 20 b.csv --> 0.864

s2_sr48power2mel260 bin fold 10 b.csv --> 0.857

s2_fpadd.csv tp + fp(inned in tp) --> 0.824

s2_sr48power2mel260 bin15 fold.csv --> 0.878

s2_tp2fp.csv --> 0.724

s2_sr48power2mel260 dif_f fold.csv --> 0.851

s2_sr48power2mel260_recfold.csv --> 0.854

s2_sr48power2mel260 bin fold 14.csv --> 0.862

s2_sr32power2mel384-ch-f0-9-f1-1_.csv --> 0.860

s2_sr32power2mel384 bin fold 15.csv --> 0.856

s2_sr48power2mel260 bin fold 16.csv --> 0.850

s2_add_fp_mark_1.csv not duplicate --> 0.416

s2__111.csv --> 0.867

s2_9515.csv --> 0.855

s2.csv --> 0.647

s2_8515.csv --> 0.858

s2.csv focal 2, 1/24 32 111 --> 0.801

s2.csv 105 --> 0.848

s2_260_bin_15_check_predict.csv --> 0.878 Best maked in kaggle, fold bin but by data

sub2_BLite4_sr32power2mel384_111_notchageSize_best_None.csv --> 0.858 diff if train in kaggle 0.870
i think is scaled

s2_260 check corrects new fold.csv --> 0.868

s2_change fold_retrain_check_corrects.csv --> 0.870

s2_260 check corrects new fold.csv --> 0.868

s2_change fold_retrain_check_corrects.csv --> 0.870

s2_change_data.csv  but fold by data bin15 this is error --> 0.839

s2_change_fold.csv bin15 by df, 32mel2384-f111 --> 0.870


==================================
RESULT:

-Cloud
  all folds, by label
  1. s2_sr48power2mel260 --> 0.874(checked get eq result on kaggle)
  2. s2_sr32power2mel384-ch-f0-9-f1-1.csv --> 0.860

  fold change by bin 15:
  1. s2_sr48power2mel260 bin15 fold.csv --> 0.878 BEST !!! by data NOT DF
  s2_260_bin_15_check_predict.csv --> 0.878 Best maked in kaggle, fold bin but by data

  2. s2_sr32power2mel384_ch_f0.9_f1.1.csv --> 0.870 (bin fold by df f-111) by DF NOT BY DATA

  fold 0
  - s2_sr32power2mel384-ch-f0-9-f1-1-pow_FOLD_0.csv --> 0.743
  - s2_full-img-sr48power2mel260_FOLD_0.csv --> 0.806

-desktop
  1.sub2_BLite4_sr48power2mel260_notchageSize_best_None.csv --> 0.841

  bin
  data  
  

  2.sub2_BLite4_sr32power2mel384_notchageSize_best_None.csv --> 0.860 fold label
  2.sub2_BLite4_sr32power2mel384_111_notchageSize_best_None.csv --> 0.858
  ---diff if train in kaggle 0.870 i think is scaled(bin fold by df)---




dublicate tp*2 labels sr32power2mel384, s2_duplicate.csv --> 0.873

==================================

26.01.2020, commited
sub2_BLite4_sr32power2mel384_notchageSize_folddata_best_None.csv --> 0.854

sub2_BLite4_sr32power2mel384_notchageSize_folddf_best_None.csv --> 0.862

sub2_BLite4_sr48power2mel260_notchageSize_folddf_best_None.csv --> 0.828

sub2_BLite4_sr48power2mel260_notchageSize_folddata_best_None.csv --> 0.859

sub2_BLite4_sr48power2mel260_notchageSize_best_None.csv --> 0.828

kaggle
s2_260_bin_15_check_predict.csv --> 0.878
s2_change fold_retrain_check_corrects.csv sr32384 --> 0.870

==================================

find best linear layers batchnorm and dropout 

FOLD = 0

s2_984_dffold_baseline fold 0.csv --> 0.818
optuna get best:

s2_meta.csv --> 0.755 add meta not optune

s2_train 8.csv -->  0.718
s2_train7.csv --> 0.807
s2_train6.csv --> 0.772
s2_train5.csv ---> 0.764
s2_train4.csv -->  0.773
s2_train3.csv --> 0.779
s2_train2.csv --> 0.681
s2_train_1 fold 0.csv --> 0.722
s2_meta_nn.Dropout(p0.1).csv --> 0.782
s2_nn.Dropout(0.21).csv train4fold0 --> 0.832
s2_meta.csv --> 0.784
s2_train4 retest.csv --> 0.773
s2_nn.Dropout(0.3) train7.csv --> 0.772
s2_resize to train 384384.csv --> 0.751
s2_train14experemet_datatestmb_motcorrect.csv --> 0.768
s2_train9.csv --> 0.773
s2_batch1d.csv --> 0.752
s2_meta2 30 epoch same change.csv --> 0.717
s2_metav2_res34.csv --> 0.744
s2_train 8.csv --> 0.718
s2_train7.csv --> 0.807
s2_train6.csv --> 0.772
s2_meta.csv --> 0.755
s2_train5.csv --> 0.764
s2_train4.csv --> 0.773
s2_train3.csv --> 0.779
s2_train2.csv --> 0.681
s2_384_dffold_Train_1 fold 0.csv --> 0.722
s2_train15.csv --> 0.759
s2_train14.csv --> 0.775
s2_train 13 16 e.csv --> 0.740
s2_train 13.csv --> 0.815
s2_dropout first 0.1 train 10.csv --> 0.758
s2_all dropout 0.1 train 10.csv --> 0.719
s2_train 12.csv --> 0.747
s2_train 11.csv --> 0.739
s2_train10.csv --> 0.773
s2_nn.Dropout(0.3) train7.csv --> 0.772
s2_v2 by each 5 image 6 loop 1 image.csv --> 0.677
s2_384 train make test 384 TTA 5.csv--> 0.767
s2_train16.csv --> 0.794
s2_train18.csv --> 0.753
s2_train17.csv --> 0.759
Use for Final Score
s2_train 18 change drop.csv -->0.703
s2_train17 change drop.csv --> 0.829

260 image size
s2_train8.csv --> 0.724
s2_train7.csv --> 0.753
s2_train6.csv --> 0.768
s2_train5.csv --> 0.776
s2_train4.csv --> 0.767
s2_train10.csv --> 0.736
s2_train9.csv --> 0.771
s2_train 3.csv --> 0.842
s2_train2 change save and data.csv --> 0.760
s2_train1 change save and data.csv --> 0.774
s2_train2.csv --> 0.760
s2_train1.csv --> 0.787


if not use loss to save best score we get best correct by its bad
s2_train 3 retast with loss.csv --> 0.736

all fold
s2_train3 all fold.csv --> 0.847 not stable



s2_fold by bin 386376 change imge.csv --train14, one fold-->0.818 experement
nn.Dropout(0.21).csv train4fold0 --> 0.832
all fold
s2_train 13 30 epoch all fold.csv --> 0.838
s2_default train 7 all fold.csv --> 0.824

s2_train 17 dropot 0.29 all fold.csv --> 0.818

s2_train14 full.csv --> 0.844
s2_train4 dropout 0.21 all fold.csv --> 0.847
s2_full fold 384376.csv --> 0.857 experement

meta
s2_meta2 16ep.csv --> 0.824


tta
s2_TTA 6 change size 384 376.csv --> 0.816




need check 260
s2_train 3 add layers with loss all fold.csv, 260 --> 0.852

s2_experement_rand_fold0_260.csv -->0.854
s2_rand_train3 default.csv --> 0.801





best


идей много и не знаю за что браться

- я увидел что   n_fft=2048, hop_length=512 не только степень двофки
  1.    n_mels = 128   n_fft = 1024    hop_length = 313  

- ablumentation ??
карочь ща сделается 111 без дбб я ее предиктну тк она меньшего размера
затем на 1 вом фолде прогоняю
- пустой безлайн
  -добавляю аргоментацию



- и вот вопрос есть еще применять так сказать к напай и потом применять спектрограмму
кек это увязать





- tune batch size



https://www.kaggle.com/hidehisaarai1213/rfcx-audio-data-augmentation-japanese-english
https://albumentations.ai/docs/
https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Blur

