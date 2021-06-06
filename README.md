<!-- https://gist.github.com/PurpleBooth/109311bb0361f32d87a2 -->
<!-- https://pandao.github.io/editor.md/en.html -->

# CBERT

Instructions for CBERT running and guidelines for installing packages.

### Dataset

Dataset: [https://drive.google.com/drive/folders/10E6nOXhRERhAmRVWla5i99jeUGWmI-Xl?usp=sharing](https://drive.google.com/drive/folders/10E6nOXhRERhAmRVWla5i99jeUGWmI-Xl?usp=sharing)

Download and extract NVD dataset and keep it in the SambaNova directory.

### Virtual environment if Anaconda is available in the system
Check your system if Anaconda module is available. If anaconda is not available install packages in the python base. If anaconda is available, then create a virtual enviroment to manage python packages.  

1. Load Module: ```load module anaconda/version_xxx```
2. Create virtual environment: ```conda create -n CBERT37 python=3.7```. Here python version 3.7 is considered.
3. Activate virtual environement: ```conda activate CBERT37``` or ```source activate CBERT37```

Other necessary commands for managing enviroment can be found here : [https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands)

### Installation of pacakages
The installations are considered for python version 3.7

Most python packages are intalled using ```pip``` or ```conda``` command. For consistency it's better to follow only one of them. If anaconda not available install packages in python base using ```pip``` command.

#### Pytorch  - 1.4.0 (or latest version)
Lastest Pytorch Version will work, for this experiment pytorch 1.4.0 is used.
Link of Pytorch installation is here: [https://pytorch.org/](https://pytorch.org/).
If Pytorch is already installed then this is not necessary.

<!--
#### Installation of Tensorflow
Only some functionalities of tensorflow is used in the project. If tensorflow is not available in the system, I will try to replace those with another function. Any version of tensorflow will do.

[https://www.tensorflow.org/overview/](https://www.tensorflow.org/overview/)
-->
#### Installation of BERT libraries, Transformers v4.5.1 (or latest version)

We will be using HuggingFace ([https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)) library for transformers.

```
pip install transformers==4.5.1
```

#### Installation of Pytorch-lightning v1.2.4 (or latest version)

We use Pytorch Lightning library to implement CBERT. Details of this library can be found, ([https://www.pytorchlightning.ai/](https://www.pytorchlightning.ai/)).

```
pip install pytorch-lightning==1.2.4
```


#### Other libraries

Install Numpy (latest version)
Command:  ```pip install numpy```
More details can be found here, [https://numpy.org/install/](https://numpy.org/install/)

Install Pandas (latest version)
Command: ```pip install pandas```

[https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html)



```
pip install sklearn
pip install wget
pip install ipywidgets
```
Package `ipynb ` for calling one python functions from another Jupyter notebook file
```
pip install ipynb
```
## Running

Execution of CBERT contains two steps. First, we run the ```1-CBERT-Pretraining``` code to pre-train the model and save it in the preferred directory. Second, we run the ```2-CBERT-Link-Prediction``` to link CVEs with CWEs.

You can run the jupypter notebook directly ```.ipynb``` or can run bare python ```.py``` using appropriate commands as arguments.

The provided code offters execution for, ```temporal, random``` partition of dataset of the CVEs. There is an additional small dataset ```dummy``` is also added to test, debug and understand the overall flow of the approach. By, default ```dummy``` dataset will run.

For jupyter notebook, change the configurations (e.g. epochs, dataset, learning rate, link numbers, etcs. ) appropriately.  List of parameters can be found inside. 


Example usage,

-- Run the jupyter note book for  pretraining ```1-CBERT-Pretraining.ipynb```

or,

```
python 1-CBERT-Pretraining.py --pretrained='bert-base-uncased' --num_gpus=2 --parallel_mode='dp' --epochs=30 --batch_size=32 --refresh_rate=200 --rand_dataset=='random'
```

The ```DP```- is for Data Parallel,  ```DDP``` is for  Distributed Data parallel.


-- Link Prediciton ```2-CBERT-Link-Prediction.ipynb```

or,

```
python 2-CBERT-Link-Prediction.py --pretrained='bert-base-uncased' --use_pretrained=True --use_rd=False --checkpointing=True --rand_dataset='random'  --performance_mode=False --neg_link=128  --epochs=25 --nodes=1 --num_gpus=2 --batch_size=64
```

## Authors
Anonymous
