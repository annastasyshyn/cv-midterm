# cv-midterm
DL CV course Mid-Term assignment on Multi-Object Tracking

## Instructions:
1. Download dataset from here:

https://github.com/VisDrone/VisDrone-Dataset.git

2. Launch ROIByteTrack:
To lanch go to: src and write your params in the main part

### Remark:
If you want to investigate code: go to "task1_2/train.ipynb" where you can see how functions work


## Train metric learning, self-supervised

```bash
cd scr
python eval.py --config-name=metric-learning/metric-learning-starting ## with the .yaml file name

python eval.py --config-name=self-supervised/self-supervised-starting ## with the .yaml file name


```