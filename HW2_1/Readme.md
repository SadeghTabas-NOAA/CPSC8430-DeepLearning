### HW02- Video Caption Generation

In order to run the codes just clone the repository and run 
    hw2_seq2seq.sh datasetDirectory output/output.txt

### Requirments:
In order to train the model the train and test datasets should be provided in the "MLDS_hw2_1_data" directory.

In order to test the model, the "training_label.json" should be provided in the "MLDS_hw2_1_data" directory as well as test dataset. 

### Run on Palmetto:
Please first request a session on a GPU node and install the following packages on palmetto:

conda create -n pytorch_env pip python=3.8.3

source activate pytorch_env

conda install pytorch torchvision cudatoolkit=9.2 -c pytorch

conda install pandas

conda install scypy

Please contact me if there is any problem. 

E-mail: sadeghs@clemson.edu 

Phone: 8034638476
equesst 
Thanks,
Sadegh,
