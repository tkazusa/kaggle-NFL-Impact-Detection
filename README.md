# kaggle-NFL-Impact-Detection

## Preparation
```
> pip install kaggle
> mkdir ~/.kaggle
> cp .kaggle.json ~/.kaggle/
> chmod 600 ~/.kaggle/kaggle.json
> kaggle competitions download -c nfl-impact-detection
> makdir data/input
> unzip nfl-impact-detection.zip -d ./data
```

#### Install ffmpeg for visualization

```
> sudo apt install -y ffmpeg
```

#### Build Detectron2 from Source
Details are on the [document](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).
```
> python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```