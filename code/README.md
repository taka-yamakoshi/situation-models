## Prediction
1. Specify the directory to output results  
Specify the output directory by setting the environmental variable `MY_DATA_PATH`  
  ```{python3}
  export MY_DATA_PATH='YOUR PATH TO DATA FILES'
  ```
2. Evaluate models  
To evaluate the performance of a model (e.g. albert-xxlarge-v2) on the dataset (e.g. `original.csv`), run  
  ```{python3}
  python wsc_prediction.py --model albert-xxlarge-v2 --dataset combined --stimuli original 
  ```


## Intervention
1. Specify the directory to output results  
Specify the output directory by setting the environmental variable `MY_DATA_PATH`  
  ```{python3}
  export MY_DATA_PATH='YOUR PATH TO DATA FILES'
  ```

2. Run causal intervention (for the `context cue` condition)  
Run the following, by replacing REP_TYPE and POS_TYPE with the representation and the position of target (e.g. `value` and `masks`).
  ```{python3}
  python wsc_intervention.py --model albert-xxlarge-v2 --dataset combined --stimuli original --rep_type REP_TYPE --pos_type POS_TYPE
  ```

3. Run causal intervention (for the `syntax cue` condition)  
Run the following, by replacing REP_TYPE and POS_TYPE with the representation and the position of target (e.g. `value` and `masks`).
  ```{python3}
  python wsc_intervention.py --model albert-xxlarge-v2 --dataset combined --stimuli control --mask_context --rep_type REP_TYPE --pos_type POS_TYPE
  ```