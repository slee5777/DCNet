# DCNet
Computer vision for differential cell counts in cytopathology images (to be linked to Scientific Report)

## Set-up
1. In order to run this notebook, please follwing the fast.ai environment setup instruction.
https://docs.fast.ai/#Installing

2. git clone this repository

3. download images (.jpg and .pkl), annotations (.txt) and model weighted (.pth) from [Google Drive](https://drive.google.com/drive/folders/1po7ZyJnQT2py3mxvpgON58H1D_hZQ19n?usp=sharing).

4. arrange file structure as follows:
```
\DCNet\(notebook.ipynb)
      \(annotations.txt)
      \(images.pkl)
      \model\(model weight.pth)
```


## Data

All the data and models used for this project can be found here:
```
Full data + models: https://drive.google.com/file/d/1JxSfyzxZlqUtoN_JPUzTzWq7_qPj95zw/view?usp=sharingData Only (Cytospin + KDSB): https://drive.google.com/file/d/1Mx11mSoGq-pYkivzayG0hvSedMBsKRBG/view?usp=sharingMask-RCNN (ablation model): https://drive.google.com/file/d/1REq4UUfKk3tuKn7Ks42xHY18IjedR7_Y/view?usp=sharing
```

Download and unzip the data folder to the root of this project directory
