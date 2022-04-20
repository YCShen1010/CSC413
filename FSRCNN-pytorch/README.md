# FSRCNN

This repository is implementation of the ["Accelerating the Super-Resolution Convolutional Neural Network"](https://arxiv.org/abs/1608.00367).


## Requirements

- PyTorch 1.0.0
- Numpy 1.15.4
- Pillow 5.4.1
- h5py 2.8.0
- tqdm 4.30.0

## Train

- Selecting and download dataset we provid: [91-image](https://drive.google.com/drive/u/0/folders/1dGpaPHnPqEzSni5rtjtfGpiFhMWwdZ4_), [Set5](https://drive.google.com/drive/u/0/folders/13StFFSeqH_EOnpe6mWT56IX8ZHlBe1d0), or [Urban-100](https://drive.google.com/drive/u/0/folders/1qDgNRVObeGh46B347GYSeR15m0D-KUZ8)

- Otherwise, you can use `prepare.py` to create custom dataset by convering to HDF5.

### How to run

```
python train.py --train-file "BLAH_BLAH/91-image_x2.h5" \
                --eval-file "BLAH_BLAH/Set5_x2.h5" \
                --outputs-dir "BLAH_BLAH/outputs" \
                --scale 1 \
                --lr 1e-4 \
                --batch-size 16 \
                --num-epochs 100 \
                --num-workers 8 \
                --seed 123                
```



## Test

- Find the weights file under `BLAH_BLAH/output/x2/best_psnr.pth`
- Test by applying your own image and to see the result

### How to run
```
python test.py --weights-file "BLAH_BLAH/output/x2/best_psnr.pth" \
               --image-file "data/your_image.bmp" \
               --scale 2
```
