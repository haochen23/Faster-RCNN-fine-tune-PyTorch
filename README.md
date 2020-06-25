# Fine-Tune Faster-RCNN on a Custom Bealge Dataset using Pytorch

## Usage

__Train the model__
```shell
python3 train.py
```
The trained model will be saved in the `output/` with name `faster-rcnn-beagle.pt`

__Model Inference__

```shell
python3 predict.py --image path/to/test/image

#for example
python3 predict.py --image beagle/val/00000184.jpg
```
