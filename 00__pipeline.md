# Hardware
[rod](IMG_2108.jpeg)

# Steps
## Collecting the data
realtime data filtering
```
/home/juu/Documents/robot_arm_record/02_01_visual_realtime.py
```

collecting the data under instruction

```
/home/juu/Documents/robot_arm_record/00_4_data_collection.py
```


## Converting to csv
converting .db3 file to csv
```
/home/juu/Documents/robot_arm_record/01_convert_db3_to_csv.py
```

## Preprocessing the data
 filtering the raw data use butterworth lowpass

```
/home/juu/Documents/robot_arm_record/03_00_smooth_data.py
/home/juu/Documents/robot_arm_record/03_01_smooth_wrench.py
```

## Labeling the data
manually choose and label the desired data
```
/home/juu/Documents/robot_arm_record/04_06_choose_data.py
```

## Training the network
Using GRN to train the data 

```
/home/juu/Documents/robot_arm_record/05_01_train_network_selected_data
```

## Verifying the network
Verify the network using online data
```
/home/juu/Documents/robot_arm_record/06_01_online_predict.py
```