# PAC Prediction Sets Under Label Shift

Code adapted from public [GitHub repository](https://openreview.net/pdf?id=DhP9L8vIyLc).

## Preparation
- Download [pretrained models](https://drive.google.com/file/d/1GPknHsdDXtrSQz2njgHmP5YCIAdHNCWL/view?usp=sharing) and unzip them (models should be placed inside the './snapshots_models' folder).

- Please check the 'requirements.txt' file for environment requirements. Install them with the following command if necessary: 
```
pip install -r requirements.txt
```
- To prevent potential instability due to different library versions, we also provide our conda environment information in 'environment.yml' for reliable reproduction.


### Data Preparation
- The CIFAR-10 and AGNews datasets will be downloaded automatically the first time you run the code.

- Download the [CDC Heart dataset](https://drive.google.com/file/d/1RRe5x9N1FVe9_Pf1nhWcbGWgQ3rpGiRu/view?usp=sharing) to the './data' folder, unzip it with the following command:
```
cd data
unzip cdc_heart.zip
```
Then run the preprocess script to preprocess the data
```
cd cdc_heart
./cdc_process.sh
```

- Download [ChestXray dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC) and unzip it into a folder containing x-ray images. Provide the image folder path to the data loader in the file main_cls_chest.py, specifically in rows 26.

Since the ChestXray dataset is a multilabel dataset, we filter and keep only the single-label samples for our problem. Please also download the [filtered single-label list](https://drive.google.com/file/d/1cXQLCEyvfKQdNeNbTL56ntfKdZz3PtZI/view?usp=share_link) and unzip it in the './data' folder.

- Entity-13 Dataset
Please download the ImageNet dataset by yourself first. Then run the following commands:
```
git clone https://github.com/MadryLab/BREEDS-Benchmarks.git
mkdir YOUR_IMAGENET_PATH/imagenet_class_hierarchy
mv BREEDS-Benchmarks/imagenet_class_hierarchy/modified/*  YOUR_IMAGENET_PATH/imagenet_hierarchy/
rm -rf BREEDS-Benchmarks
```

## Run Prediction Sets under Label Shift and Baselines
For each dataset, simply run the corresponding script, e.g.:

```
./script/run_main_cls_cifar10.sh
./script/run_main_cls_chest.sh
./script/run_main_cls_heart.sh
./script/run_main_cls_agnews.sh
./script/run_main_cls_entity.sh
```

You can adjust the epsilon and delta values in each script. The results will be saved in the './snapshots' folder.

## Plotting the Error-Size Figures
Run 
```
./script/run_plot.sh
```
if you have run experiments for all datasets. Otherwise please the pick the corresponding python command for your dataset.