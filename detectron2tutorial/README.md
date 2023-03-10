### Run a prediction with pre-trained model

```bash
> wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O input.jpg
> python predict_with_pretrained_model.py
```

### Data preparation

```bash
> python data_preparation.py
```


### Run a training on a custom dataset

```bash
> wget https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip
> unzip balloon_dataset.zip > /dev/null
> python train.py
```

### Run a prediction and an evaluation

```bash
> python predict_evaluation.py
```