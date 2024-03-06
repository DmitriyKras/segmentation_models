# segmentation_models

This repository contains python scripts for training, validation and onnx exporting different semantic segmentation models. All models designed only for single class segmentation (person).
Changing number of classes is forbidden. Allowed models:


- [EfficientUnet](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w22/Baheti_Eff-UNet_A_Novel_Architecture_for_Semantic_Segmentation_in_Unstructured_Environment_CVPRW_2020_paper.pdf)
- [LiteSeg](https://github.com/tahaemara/LiteSeg)
- [BiSeNet](https://github.com/ooooverflow/BiSeNet)
- [ESNet](https://github.com/xiaoyufenfei/ESNet)


## Training

*train.py* - python script for model's training. Run with command line:

`python3 train.py --model [model name] --loss [loss function name] --img [input image size] --batch-size [number of images per batch] --epochs [number of epochs to train for] --augment [whether to use augmentation] --save-period [period in epochs to save checkpoint] --patience [number of epochs to stop training if no improvement observed] --data [path to data.yaml with dataset structure] --lr [learning rate] --decay [coef for exponential lr decay]`

`--model [effunet, esnet, bisenet, liteseg_shufflenet, liteseg_mobilenet]`
`--loss [dice, tversky]`
`--augment [True, False]`

`--data` data.yaml contains:

path: absolute path to dataset root

train: relative path from dataset root to jpg (images) for training

val: relative path from dataset root to jpg (images) for validation

test: relative path from dataset root to jpg (images) for testing (required for `--task test` in `val.py`)

Folder with segmentation masks must be in same directory as folder with images and have name masks. For example (train/images и train/masks).
Segmentation masks must have same suffix and name as corresponding images (jpg). Pixels with objects contain 255, pixels with background contain 0.

Script saves models weights in weights/modelname and training results in logs/modelname 

## Validation

*val.py* - script for validation and testing. Run with command line:

`python3 val.py --model [model name] --img [input image size] --batch-size [number of images per batch] --data [path to data.yaml with dataset structure] --task [val or test] --weights [path to pt weights]`

`--task [val, test]` - uses val or test folders from data.yaml.

Script saves validation/test results in logs/modelname.

## Export

*export.py* - script for onnx export. Run with command line:

`python3 export.py --model [model name] --img [input image size] --weights [path to pt weights] --onnx-file [path to onnx file] --opset [opset version] --checking [whether to perform onnx inference]`

`--checking [True, False]`

Script performs onnx exporting with onnxsim simplifying. `--checking` performs inference with onnxruntime on test.jpg image then on test_video.mp4. This option requires test.jpg and test_video.mp4.
