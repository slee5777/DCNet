# DCNet
Computer vision for differential cell counts in cytopathology images (to be linked to Scientific Report)

## Setup
1. In order to run this notebook, please follwing the fast.ai environment setup instruction.
https://docs.fast.ai/#Installing

2. git clone this repository

3. download images (.jpg and .pkl), annotations (.txt) and model weighted (.pth) from [Google Drive](https://drive.google.com/drive/folders/1po7ZyJnQT2py3mxvpgON58H1D_hZQ19n?usp=sharing).

4. Before running the notebook, arrange files as follows:
```
\DCNet\(notebook.ipynb)
      \(annotations.txt)
      \(images.pkl)
      \model\(model weight.pth)
```

## Model Architecture
<img src="fig7_DCNet_arch.png" width="500">

## Schematic diagram
see dcnet-resnet34-entire-model.onnx.svg

## Supplementary information

All the data and models used for this project can be found here:
```
Full data + models: https://drive.google.com/file/d/1JxSfyzxZlqUtoN_JPUzTzWq7_qPj95zw/view?usp=sharing
Data Only (Cytospin + KDSB): https://drive.google.com/file/d/1Mx11mSoGq-pYkivzayG0hvSedMBsKRBG/view?usp=sharing
Mask-RCNN (ablation model): https://drive.google.com/file/d/1REq4UUfKk3tuKn7Ks42xHY18IjedR7_Y/view?usp=sharing
```

## Disclaimer
https://www.freeprivacypolicy.com/live/5a5dff6f-592a-4271-8fcc-8dbf729d1171
