# Argument Parser 

### `split.py`


### `run.py`
````
--data_dir_augmented /data/users/etosato/ANM_Verona/data/FCmaps_augmented_processed \
--data_dir /data/users/etosato/ANM_Verona/data/FC_maps_processed \
--split_csv /data/users/etosato/ANM_Verona/data/ADNI_PSP_splitted.csv \
--group1 ADNI \
--group2 PSP \
--checkpoints_dir /data/users/etosato/ANM_Verona/src/cnn/checkpoints \
--checkpoint_path /data/users/etosato/ANM_Verona/src/cnn/checkpoints/best_model.pt \
--plot_path /data/users/etosato/ANM_Verona/plots/loss_curve.png \
--model_type resnet \
--epochs 2 \
--batch_size 4 \
--crossval_flag \
--evaluation_flag
```