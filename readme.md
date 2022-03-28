# IceBerg: Measure Distrust in Relational Database [Scalable Data Science]

****

## Implementation Details

This is Tensorflow solution in Python3.7.4 that has been implemented and tested against Tensorflow 2.3.0, numpy 1.17.2, sklearn 0.21.3, pandas 0.25.1.

## Appendices
The appendix provides the constraint rules found by our algorithm in two publicly available datasets.

The constraint is of the form : 

$$	P_1 \wedge P_2 \wedge \ldots \wedge P_m \Rightarrow C_1 \wedge C_2 \wedge \ldots \wedge C_n$$


Here, $P_i$ is a premise condition and $C_i$ is a conclusion condition. Each condition is in the format of $A_i=v$, $A_i>v$, $A_i<v$ or $A_i\in [v_1,v_2]$.  Note that we allow the attribute in $C_i$ to be virtually constructed from existing attributes in the relational table.

##Input Data
The datasets are saved in the [Datasets folder](./datasets/.).

**[WBC](./datasets/wdbc.csv)**: Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.

**[Lymphography](./datasets/lymphography.csv)**: This is one of three domains provided by the Oncology Institute that has repeatedly appeared in the machine learning literature.

## Test Scripts
We provide the trained model as well as the tuned parameters.
The path where the models are located is [models](./code/models/LYM_18_18) and the script is [KD-VAE.py](./code/KD-VAE.py).

When test, the --pt parameter is set to **False** and the --eval parameter is set to **True**.

After the above configuration environment is set up, the user only needs to run [KD-VAE.py](./code/KD-VAE.py) to get the constraint rules under the WBC dataset.
## Parameter setting

To facilitate user testing, we have included the path . /code/parameter_setting, the user can open the file named after the dataset and copy its contents directly, replacing the code under the main function.

- [WBC dataset parameter setting](./code/parameter_setting/WDBC_parameter_setting.txt)

- [Lymphography dataset parameter setting](./code/parameter_setting/LYM_parameter_setting.txt)


## Pre-training

Since the overall model is too complex and training it together causes difficulty in convergence and inefficiency, two-part training is used to train a portion of the parameters first and then train the remaining parameters of the model with this portion fixed.

###Introduction of hyperparameters
+ data_filename:Dataset file path (./datasets/wdbc.csv/ | ./datasets/lymphography.csv)
+ map_size：size of map
+ model_type：choose a model (WBC/LYM)
+ x latent size: size of input embedding
+ rnn_size:size of RNN hidden state
+ neg_size:size of negative sampling
+ num_epochs:number of epochs
+ grad_clip:clip gradients at this value
+ learning_rate:learning rate
+ decay_rate:decay of learning rate
+ batch_size:minbatch size
+ model_id:model id
+ partial_ratio:partial tuple evaluation
+ eval:partial tuple evaluation
+ pt:partial tuple evaluation

When pre-training, first specify the ***data filename*** parameter to select the dataset, and secondly, to distinguish between different models, the ***model_type*** parameter designs the name according to your preference.The ***pt*** parameter is set to **True** and the eval parameter is set to **False** to run the model to start training.

## training
To start training, you need to set the -pt parameter and --eval to **False** and run the program.

## evaluating
To start evaluating, you need to set the -pt parameter to **False** and set the --eval parameter to **True** and run the program.


## Visualization
![image](https://github.com/huwentao0579/supplementary_file_1/blob/master/constraint1.gif)
![image](https://github.com/huwentao0579/supplementary_file_1/blob/master/constraint2.gif)
![image](https://github.com/huwentao0579/supplementary_file_1/blob/master/constraint3.gif)
