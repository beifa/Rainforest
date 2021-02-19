Experiment 1:

    data: cut data(by times and drop frec by max/min) instant load
    param:
        key_fold_v2_older veresion
        size = (128, 550)
        nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pos_weights)
        optim.Adam(model.parameters())
        ReduceLROnPlateau(optimizer, 'min')
        SAMPLERATE = 48000
        batch_size = 14
        n_workers = 4
        epoch = 30
    result:
        test_train_nn_subtozeros_res50_cat_instaload_sigm_30allfoldyes_div5.csv --> 0.744
        test_train_nn_subtozeros_res50_cat_instaload_sigm_30allfoldno_div5.csv --> 0.744

        test_train_nn_subtozeros_res50_cat_instaload_max_30allfoldyes_div5.csv --> 0.727
        test_train_nn_subtozeros_res50_cat_instaload_max_30allfoldno_div5.csv --> 0.727

Experiment 2:

    train > 8h
    data: 
    param:
        epoch = 40
    result:
    test_train_nn_subtozeros_res50_cat_instaload_max_40allfoldno_div5.csv --> 0.659
    test_train_nn_subtozeros_res50_cat_instaload_sigm_40allfoldno_div5.csv --> 0.684

    test_train_nn_subtozeros_res50_cat_instaload_max_40allfoldyes_div5.csv --> 0.659   
    test_train_nn_subtozeros_res50_cat_instaload_sigm_40allfoldyes_div5.csv --> 0.684


Experiment 3:

    train epoch ~ 1 min, predict 2min
    Res50_128_30_sfold_Plateau_Adam_pos_weights_lr3e-5_scaled_best_sfold
    best:
        sub2_Res50_128_30_sfold_Plateau_Adam_pos_weights_lr3e-5_scaled_best_sfold.csv --> 0.624
        sub_Res50_128_30_sfold_Plateau_Adam_pos_weights_lr3e-5_scaled_best_sfold.csv --> 0.630
    final:
        sub2_Res50_128_30_sfold_Plateau_Adam_pos_weights_lr3e-5_scaled_final_sfold.csv --> 0.646
        sub3_Res50_128_30_sfold_Plateau_Adam_pos_weights_lr3e-5_scaled_final_sfold.csv --> 0.656 not /5
        sub_Res50_128_30_sfold_Plateau_Adam_pos_weights_lr3e-5_scaled_final_sfold.csv --> 0.645


Experiment 3:

    gfold
    sub2_Res50_128_30_gfold_Plateau_Adam_pos_weights_lr3e-5_scaled_final_gfold.csv --> 0.641
    sub_Res50_128_30_gfold_Plateau_Adam_pos_weights_lr3e-5_scaled_final_gfold.csv --> 0.644
    sub_Res50_128_30_gfold_Plateau_Adam_pos_weights_lr3e-5_scaled_best_gfold.csv --> 0.659
    sub2_Res50_128_30_gfold_Plateau_Adam_pos_weights_lr3e-5_scaled_best_gfold.csv --> 0.662   


Experiment 4:

    kernel Res50_128_30_sfold_Plateau_Adam_pos_weights_lr_default --fold_type  

    default lr Adam
    sfold
    sub_Res50_128_30_sfold_Plateau_Adam_pos_weights_lr_default_best_sfold.csv --> 0.628
    sub2_Res50_128_30_sfold_Plateau_Adam_pos_weights_lr_default_best_sfold.csv --> 0.610
    gfold
    sub_Res50_128_30_gfold_Plateau_Adam_pos_weights_lr_default_final_gfold.csv --> 0.745
    sub2_Res50_128_30_gfold_Plateau_Adam_pos_weights_lr_default_final_gfold.csv --> 0.736

Experiment 5:

    add pr and f1
    add pr metric to shulder
    fold0 = auc_max: 0.9929819010502273 --> 0.9917281883892004, PRw: 0.7649469024964067, PRmicro: 0.7731739369855462)
    fold1 = auc_max: 0.9954190417467577 --> 0.9929032757129285, PRw: 0.8312997585451184, PRmicro: 0.8407948755758075)
    fold2 = auc_max: 0.9928479511800413 --> 0.9895558307158888, PRw: 0.8110219368815959, PRmicro: 0.8129069173651655)
    fold3 = auc_max: 0.9969022105073234 --> 0.9926124860520344, PRw: 0.9108049929656516, PRmicro: 0.8970609145357961)
    fold4 = auc_max: 0.9926975299987659 --> 0.9891143129722106, PRw: 0.8521903181750816, PRmicro: 0.8212990241364623)
    sub_Res50_128_30_gfold_Plateau_Adam_pos_weights_lr_default_nopretrain_best_gfold.csv --> 0.728
    sub2_Res50_128_30_gfold_Plateau_Adam_pos_weights_lr_default_nopretrain_best_gfold.csv --> 0.734

#---------------change idx in folds need peretest

Experiment 6:

    only tp and duplicate
    sub2_Res50_128_30_gfold_Plateau_Adam_pos_weights_lr_default_nopretrain_onlyTP_best_gfold.csv --> 0.654
    sub_Res50_128_30_gfold_Plateau_Adam_pos_weights_lr_default_nopretrain_onlyTP_best_gfold.csv --> 0.658 

Experiment 7:

    gfold and error index
    fold1 auc_max: 0.8865739089594792 --> 0.8823304665288573, PRw: 0.24453101404608846, PRmicro: 0.17398139754875094)
    fold2 auc_max: 0.9205824577538607 --> 0.9195019572317097, PRw: 0.29198340505151044, PRmicro: 0.2655693091644364)
    fold3 auc_max: 0.9278325111267698 --> 0.9265716718325508, PRw: 0.3687466438790393, PRmicro: 0.32556909798431055)
    fold4 auc_max: 0.8881359910737517 --> 0.8734365391111453, PRw: 0.2204485134140436, PRmicro: 0.14787231571047094)
    fold5 auc_max: 0.918444050473926 --> 0.9133971460750074, PRw: 0.3776763768430912, PRmicro: 0.2828740093933182)

    sub2_Res50_128_30_gfold_Plateau_Adam_pos_weights_lr_default_nopretrain_pretesterridx_best_gfold.csv --> 0.674
    sub_Res50_128_30_gfold_Plateau_Adam_pos_weights_lr_default_nopretrain_pretesterridx_best_gfold.csv --> 0.699

Experiment 8:

    gkffold by  recording_id
    fold1 auc_max: 0.8977232600555368 --> 0.8944331175791037, PRw: 0.24634686985109217, PRmicro: 0.19529752823140678)
    fold2 auc_max: 0.888403556624415 --> 0.8878448538651839, PRw: 0.3345128962669108, PRmicro: 0.2690135523829129)
    fold3 auc_max: 0.9009956779466606 --> 0.8971658609686525, PRw: 0.34352532802714025, PRmicro: 0.25977561538330995)
    fold4 auc_max: 0.9079284643396889 --> 0.9066720736461079, PRw: 0.32580567383774284, PRmicro: 0.279191248325829)
    fold5 auc_max: 0.9265137161745213 --> 0.9252739144786354, PRw: 0.33093548072657447, PRmicro: 0.27398568023485614)
    sub2_Res50_128_30_gfold_Plateau_Adam_pos_weights_lr_default_nopretrain_pretesterridx_best_gkffold.csv --> 0.669
    sub_Res50_128_30_gfold_Plateau_Adam_pos_weights_lr_default_nopretrain_pretesterridx_best_gkffold.csv --> 0.693


Experiment 9:

    pretrain
        -best from folds
        sub2_Res50_128_30_gfold_Plateau_Adam_pos_weights_lr_default_pretrain_pretesterridx_best_gfold.csv --> 0.685
        sub_Res50_128_30_gfold_Plateau_Adam_pos_weights_lr_default_pretrain_pretesterridx_best_gfold.csv -->  0.708
  

--------------  change predict loop

Experiment 10:

    olds folds group from kaggle

    change predict loop
    было что 10 нарезок предсказываличь как одна тоесть я ее передовал целиком все 10
    и эти 10 как банч оно перемножалось
    теперь каждая из 10 предсказывается и добавляется 
        prob3 = torch.max(y_, dim=0)[0]
        prob2 = torch.max(nn.Sigmoid()(y_), dim=0)[0]
        prob = torch.max(y_, dim=0)[0]
            sub.loc[name[0]] += pred 
            sub2.loc[name[0]] += pred2 
            sub3.loc[name[0]] += pred3
        sub3.loc[name[0]] /= 10

    30 epoch
    optimizer = optim.Adam(model.parameters(), lr = 0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.4, verbose=True)

    s3.csv --> 0.650
    s2.csv --> 0.679
    s1.csv --> 0.718


Experiment 11
    
    tp
    add 55 epoch
    s3.csv --> 0.623
    s2.csv -->0.668
    s1.csv --> 0.675

Experiment 12

    tp
    size image 224
    add 55 epoch
    
    s2.csv --> 0.612
    s1.csv --> 0.634

all up old fold test 10-12

change folds

Experiment 13
    stratified tp only
    optimizer = optim.Adam(model.parameters(), lr = 0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.4, verbose=True)
    s2.csv --> 0.691
    s1.csv --> 0.696
    sub1 div(5) folds (1).csv --> 0.696


-----
maked new data (224, 563), add skip librosa.power_to_db(mel_spec)
-----

Experiment 14
    change identity in res50
    adamw default 35 epoch
    s1.csv --> 0.757

Experiment 15
    adam poletau epoch 35
    s1.csv --> 0.751

Experiment 16
    18 commit
    full data, adam poletau epoch 35
    optimizer = optim.Adam(model.parameters())#, lr = 0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.9, patience=3, verbose=True)
    s1.csv --> 0.759

Experiment 17

    full, my pc           
    optimizer = optim.Adam(model.parameters(), lr = 0.001, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.9, mode='min', patience=4, verbose=True)
    sub_Res50_128_35_gfold_Plateau_Adam_pos_weights_lr_default_pretrain_best_gfold.csv -->  0.721

Experiment 18

    kfold, full
    sub_Res50_128_35_kfold_Plateau_Adam_pos_weights_lr_default_pretrain_best_kfold.csv --> 0.668


Experiment 19
    epoch 35, tp
    optimizer = optim.Adam(model.parameters())#, lr = 0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.9, patience=3, verbose=True)
    self.myfc = nn.Sequential(nn.Linear(in_ch, 1024),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2),
                                  nn.Linear(1024, 1024),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2),
                                  nn.Linear(1024, 24)
                                 )   

    s2.csv --> 0.702
    s1.csv --> 0.700


Experiment 20

    epoch 80, no good, tp 
    self.myfc = nn.Sequential(nn.Linear(in_ch, 1024),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2),
                                  nn.Linear(1024, 1024),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2),
                                  nn.Linear(1024, 24)
                                 )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.4, verbose=True)
    s1.csv --> 0.754

Experiment 21

    epoch 80,tp 
     self.myfc = nn.Sequential(nn.Linear(in_ch, 1024),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2),
                                  nn.Linear(1024, 1024),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2),
                                  nn.Linear(1024, 24)
                                 )  
    optimizer = optim.Adam(model.parameters())#, lr = 0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.9, patience=3, verbose=True)
    s1 (1).csv --> 0.685

Experiment 22 

    epoch 35, full data    
    StratifiedKFold(n_splits=5, shuffle=True, random_state=13) by species_id)
    optimizer = optim.Adam(model.parameters())#, lr = 0.001, weight_decay=0.0001)   
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.9, patience=3, verbose=True)

    sub_Res50_128_35_kfold_Plateau_Adam_pos_weights_lr_default_pretrain_default_final_kfold.csv --> 0.736
    sub_Res50_128_35_kfold_Plateau_Adam_pos_weights_lr_default_pretrain_default_best_kfold.csv --> 0.776

Experiment 22

    add to model:

        sigmoid = nn.Sigmoid()
        
        class Swish(torch.autograd.Function):
            @staticmethod
            def forward(ctx, i):
                result = i * sigmoid(i)
                ctx.save_for_backward(i)
                return result
            @staticmethod
            def backward(ctx, grad_output):
                i = ctx.saved_variables[0]
                sigmoid_i = sigmoid(i)
                return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))
        
        class Swish_Module(nn.Module):
            def forward(self, x):
                return Swish.apply(x)
        
        class Res50(nn.Module):            
            def __init__(self):
                super(Res50, self).__init__()        
                self.model = pretrainedmodels.__dict__['resnet50'](pretrained = 'imagenet')
                # list(resnet50.children())[:-2]
                in_ch = self.model.last_linear.out_features  
        #         self.myfc = nn.Linear(in_ch, 24)
        #         in_ch = list(self.model.children())[-3][-1].bn3.num_features
                self.myfc = nn.Sequential(nn.Linear(in_ch, 1024),
                                        Swish_Module(),
                                        nn.BatchNorm1d(1024),
                                        nn.Dropout(p=0.2),
                                        nn.Linear(1024, 1024),
                                        nn.BatchNorm1d(1024),
                                        Swish_Module(),
                                        nn.Dropout(p=0.2),                                 
                                        nn.Linear(1024, 24)
                                        ) 

    80 epoch, tp, stratifiedkf, species_id
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.4, verbose=True)

    NOT NEED 80 EPOCH ALL saved UNDER 30 epoch
    s1.csv --> 0.778



Experiment 23

    epoch 80, no good, fulldata 
    self.myfc = nn.Sequential(nn.Linear(in_ch, 1024),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2),
                                  nn.Linear(1024, 1024),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2),
                                  nn.Linear(1024, 24)
                                 )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.4, verbose=True)
    s1.csv --> 0.801

Experiment 24 

    epoch 50, no good, fulldata 
    change
    StratifiedKFold by test recording_id

    self.myfc = nn.Sequential(nn.Linear(in_ch, 1024),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2),
                                  nn.Linear(1024, 1024),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2),
                                  nn.Linear(1024, 24)
                                 )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.4, verbose=True)
    s1.csv --> 0.803

Experiment 24 

    50 epoch very big not good
    epoch 35, gfold by id, full data add save corrects after 2 epoch
    s1.csv --> 0.752 best correct
    s1.csv --> 0.763 best fold

Experiment 25

    epoch 35 best model not change, fold by recording
    res 34
    s1.csv --> 0.799

Experiment 26

    epoch 35 best model not change, fold by recording
    res 101
    s1 (1).csv --> 0.792

Experiment 25
    epoch 35 best model not change, fold by recording
    res se_resnet50
    s1_se_resnet50_corrct.csv --> 0.769
    s2.csv --> 0.767
    s1.csv --> 0.795

Experiment 26
    epoch 35 best model not change, fold by recording
    resnet152
    s1_resnet152.csv --> 0.783


Experiment 27
    full
    epoch 25 best model not change, fold by recording
    res50
    new data 224x563 v2
    224-563-res50
    but i stupid test predict by old data
    s2.csv --> 0.323
    s1.csv --> 0.441
    peretest
    s1.csv --> 0.776
    correct





Experiment 28
    tp
    experement cut tp,  new data 10x224x563 v3
    10-224-563-res50
    s1.csv --> 0.541


Experiment 29
    tp
    new data 10x224x563 v3
     я буду брать только одну и учится на одной (marked) 1-224-563-res50
     s1.csv 224x563, train_img_10_224, tp only --> 0.774

Experiment 30

  def __init__(self):
    super(EffB3, self).__init__()
    self.model = geffnet.create_model('efficientnet_b3', pretrained=True) 
    in_ch = self.model.classifier.in_features #1536
    self.myfc = nn.Sequential(nn.Linear(in_ch, 24))
    self.model.classifier = nn.Identity() 
    
    sub2_B3_128_35_kfold_Plateau_Adam_pos_weights_lr_default_pretrain_best_kfold.csv --> 0.731
    sub_B3_128_35_kfold_Plateau_Adam_pos_weights_lr_default_pretrain_best_kfold.csv --> 0.760

SCALED 128X563 --> 224X563========================================================================

Experiment 30
    all, 224x563 B3 all default,  SCALED 128X563 --> 224X563
    sub2_B3_224x563_35_kfold_Plateau_Adam_pos_weights_lr_default_pretrain_best_kfold.csv --> 0.712
    sub_B3_224x563_35_kfold_Plateau_Adam_pos_weights_lr_default_pretrain_best_kfold.csv --> 0.775

Experiment 31 -----------commit best curren model
    b3 only tp
    NOT LOAD 224x563 B3 all default, SCALED 128X563 --> 224X563
    very fast !!!!!!!!!!!!!
    sub_B3_224x563_35_kfold_Plateau_Adam_pos_weights_lr_default_pretrain_TP_best_kfold.csv --> 0.794

OH NOT ALL LOCAL TRAIN NOT CORRECT BECAUSE NOT CHANGE IMAG FOLDER=================================

Experiment 31

    20 epoch 
    b3 only tp
    _img(380, 1126)
    s1.csv --> 0.652

Experiment 32 peretest for correct data 

    30epoch
    b3 only tp
    LOAD 224x563 B3 all default
    very fast !!!!!!!!!!!!!
    sub_B3_224x563_35_kfold_Plateau_Adam_pos_weights_lr_default_pretrain_TP_best_kfold.csv --> 0.786
    

Experiment 33

    563*380
    b3 only tp
    train_img_(380, 1126)_npy    
    sub_B3_224x563_35_kfold_Plateau_Adam_pos_weights_lr_default_pretrain_TP_best_kfold.csv --> 0.793

Experiment 34

    tp
    380*380
    sub_B3_380x380_30_kfold_Plateau_Adam_pos_weights_lr_default_pretrain_TP_best_kfold.csv --> 0.782


Experiment 35

    tp
    224*224
    s1.csv 224*224b3 --> 0.692

Experiment 36

    tp
    img_128_563_v1
    scaled 224 try get resit Experement 31   
    correct --> 0.773
    best --> 0.777

Experiment 37

    res34 scale 128-->224 224*563 all defaul
    adam, Plateau
    in head 
    self.myfc = nn.Sequential(nn.Linear(in_ch, 24))
    TP
    30 epoch 
    s1_res34_224_scaled.csv --> 0.742


Experiment 38

    res34 380*763 all defaul
    adam, Plateau
    in head 
    self.myfc = nn.Sequential(nn.Linear(in_ch, 24))
    TP
    30 epoch 
    s1380768bres34.csv --> 0.354 very long 1.35 1 pred * 5

Experiment 39

    effb0 scale 128-->224 224*563
    adam, Plateau
    in head 
    self.myfc = nn.Sequential(nn.Linear(in_ch, 24))
    TP
    30 epoch 
    s1_EffB0_224scaled.csv --> 0.767

Experiment 40

    effb0 NOT scale 224*563
    adam, Plateau
    in head self.myfc = nn.Sequential(nn.Linear(in_ch, 24))
    TP
    30 epoch 
    s1_EffB0_224notscale.csv --> 0.737

Experiment 41

    res34 NOT scale 224*563
    adam, Plateau
    in head 
    self.myfc = nn.Sequential(nn.Linear(in_ch, 24))
    TP
    30 epoch 
    s1_res34 224563 no scaled.csv --> 0.712

Experiment 42

    effb1 scale 128, 224*563 30 adam 
    s1_effb1 scale 128 224563 30 adam.csv --> 0.761

Experiment 43

    effb1 30 adam plateau 224*563 n
    s1_effb1 30 adam plateau 224563 n.csv --> 0.781

Experiment 44

    res 50 adam plateau 30 no s 224*563
    s1_res50_notscale.csv --> 0.749

Experiment 45

    res 50 adam plateau 30 scaled 128 --> 224*563
    s1_res50_scaled.csv --> 0.755

Experiment 46

    effb2 scale 128, 224*563 30 adam 
    s1_b2_notscale.csv --> 0.798

Experiment 47

    effb2 30 adam plateau 224*563 n
    s1_b2 scaled.csv --> 0.800


Experiment 48

    tf_effb0_ns scale 128, 224*563 30 adam
    s1_tf_effb0_scale.csv --> 0.782

Experiment 49

    tf_effb0_ns 224*563 30 adam Noscaled
    s1_tf_effb0_Noscale.csv --> 0.754

Experiment 50

    tf_effb1_ns 224*563 30 adam Noscaled
    s1_tf_effb1_ns 224563 30 adam Noscaled.csv --> 0.786



    tf_effb1_ns 224*563 30 adam scaled
    s1_tf_effb1_scale.csv --> 0.754

Experiment 52
    peretest

    Experiment 46
    effb2 scale 128, 224*563 30 adam 
    s1_b2_notscale.csv --> 0.798

    sub_B2_563x224_30_kfold_Plateau_Adam_pos_weights_lr_default_pretrain_TP_best_kfold.csv --> 0.797

Experiment 53

    448*576 7500GbGPU
    sub_B2_576x448_30_kfold_Plateau_Adam_pos_weights_lr_default_pretrain_TP_best_kfold.csv --> 0.789

Experiment 54

    tf_effb3_ns 224*563 30 adam Npscaled
    s1_tf_effb3_Noscale.csv --> 0.784

    s1_b3scal.csv --> 0.782

Experiment 55
    
    tf_effb4_ns 224*563 30 adam Npscaled
    predict on 128*563???
    perertestt
    s1_b4_nos.csv --> 0.805

Experiment 56

    tf_effb2_ns 224*563 30 adam scaled
    s1_etf_effb2_scale.csv --> 0.660

Experiment 57

    s1_effb5_Nos.csv --> 0.792

Experiment 58

    s1_lite3.csv --> 0.786

Experiment 59

    s1_efficientnet_b6.csv --> 0.627

Experiment 60

    s1_tf_efficientnet_lite2.csv --> 0.782

Experiment 60
    
    'efficientnet_lite0',
    s1_lite0.csv --> 0.767

Experiment 61
    
    'efficientnet_lite1',
    s1_tf_efficientnet_lite1.csv --> 0.751

Experiment 62 -------best test 
    'efficientnet_lite4',
    s1_lite4.csv --> 0.810

Experiment 63
    s2.csv--> 0.738
    s1_lite4_fulldata.csv --> 0.757

Experiment 64
    s1_notfull_lite4.csv --> 0.783

Так не удобно так записывать проще это делать в Excel

================
lite4 5.7GB nvidia
    -1 epoch train/valid == 15 seconds
    -valid 4 min one fold

1. перетест на моем компе
    - время лучшее
        1.lite4 5.7GB nvidia
            -1 epoch train/valid == 15 seconds / 4.3 min
            data 224 v1 
            sub2_LiteB4_576x224_30_kfold_Plateau_Adam_pos_weights_lr_default_pretrain_TP_best_kfold.csv --> 0.815            
            sub_LiteB4_576x224_30_kfold_Plateau_Adam_pos_weights_lr_default_pretrain_TP_best_kfold.csv --> 0.792  
            
            old 224
            sub2_LiteB4_576x224_30_kfold_Plateau_Adam_pos_weights_lr_default_pretrain_TP_best_kfold.csv --> 0.809        
            sub_LiteB4_576x224_30_kfold_Plateau_Adam_pos_weights_lr_default_pretrain_TP_best_kfold.csv --> 0.793

        2.tf_b4
        3.effb2

2. оставить лучшую скор \ время
3. попытаемя найти лучшие пераметры для картинок
    - мел
    - тиме
    - фраме
    и тд.

original 0.9 1.1
    f_min, f_max = df.f_min.min() * 0.8, df.f_max.max() * 1.2

================


Experiment 65
stack image by pandas competition image size 563*640
all default no resaize, tp, 30

1. predict default
2. predict experement stak by train
    s2.csv --> 0.679
    s1_experement_mergeimge.csv --> 0.687

Experiment 66
crop
tp, 30, 10 sec
test = torch.Size([1, 11, 3, 128, 938])
s2_crop default.csv --> 0.691

Experiment 67
crop v1
tp, 30, 6 sec
test = torch.Size([1, 19, 3, 128, 563])
s2_cropv1.csv --> 0.681

Experiment 68
corp v2
s2_random-crop.csv --> 0.638

Experiment 69
corp v3

s2_crop.csv --> 0.696