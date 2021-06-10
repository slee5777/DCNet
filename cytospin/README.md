# Nuclei project

## Summary

Develop a data pipeline for digialized cytospin slides to solve classification and segmentation problems.

I)   Use LabelBox to annotate CenterPoints by assessors (see A)
II)  Download the labels from I) and then let a expert-in-the-loop (consensus method) to generate ground truth CenterPoints (6b_GroundTruth_Consensus.ipynb)
III) Convert ground truth CenterPoints into polygons automatically and then upload them back to LabelBox (F_CenterPoints_to_Polygons-WIP.ipynb)
IV)  Labelers modify polygons into 4-extremity points
V)   Run DEXTR to generate segmentation masks

TODO:
a)   Clean up CenterPoints prediction masks into points (or small cycles)
b)   Use U-net to train on segementation masks - easy
c)   Codes update due to LabelBox changing JSON format.

### Segementation Experiments:

1. Use labelbox to annotate bounding boxes or extreme points
2. Download annotations and run DEXTR or PolygonRNN to generate segmentation labels
  
Current state: Out-of-the-box DEXTR algorithms outperformed PolygonRNN.  (Note: Model weight from PolyRNN++ is not available.)

#### Bounding Boxes + PolygonRNN:  
1. Start with the labelbox project: [Nuclei Bounding Box](https://app.labelbox.com/projects/ck2y7wl63x4el0811jl9oyfht/overview)  
We use this project to human annotate bounding boxes
2. We then download these bounding box annotations `3_PolygonRNN_API.ipynb` and run the polygonrnn algorithm on it.
3. We take these predicted segmentations and upload them to a separate labelbox project - [AutoPolyrnn](https://app.labelbox.com/projects/ck4rzt4fgq8by08476ocsj8bs/overview)
4. This auto uploaded segmentations are then sent to humans for corrections


#### Extreme Points + DEXTR:  
1. Start with the labelbox project: [ExtremePoints Polygon](https://app.labelbox.com/projects/ck4rtdrcvdfer0870n0im528k/overview)  
We use this project to human annotate extreme points. These are basically 4-point polygons
2. We then download these extreme point annotations `4_Dextr_API.ipynb` and run the dextr algorithm on it.
3. We take these predicted segmentations and upload them to a separate labelbox project - [AutoDextr](https://app.labelbox.com/projects/ck4s6bg2531x608223qxgwxmc/overview)
4. This auto uploaded segmentations are then sent to humans for corrections


## A. Environment Setup
1. `git@github.com:bearpelican/nuclei.git`

    SL: run `conda config --set pip_interop_enabled True` first, to avoid pip install failure using environment.yml file.
2. `conda env create -f environment.yml`

3.  To install OpenSlide (www.openside.org) for tiling .svs files in Pawsey (Ubuntu environment)
     `(sudo) apt-get install python3-openslide` in base
     activate nuclei environment, then `pip install openslide-python`.  
     Note: There is a known problem if setuptools version > 46 (see https://github.com/pypa/setuptools/issues/2017).  Run `conda install setuptools=45` first.

## To run polygonrnn (Note: better to use DEXTR below)

Run the polygonrnn [notebook](./notebooks/labelbox/3_PolygonRNN_API.ipynb)


You'll neeed to clone the polyrnn project. This git project has been updated to Pytorch 1.3:
```
cd perth-2020
git clone git@github.com:bearpelican/polyrnn.git
```

You'll also need to download the pretrained models:
```
Download link:
https://urldefense.proofpoint.com/v2/url?u=http-3A__www.cs.toronto.edu_polyrnn_code-5Fsignup_download.php-3Fkey-3DdZi89AuhYb&d=DwIBAw&c=qgVugHHq3rzouXkEXdxBNQ&r=iJQkRGiQHXIoaZ_jxElRSA&m=xwvhr2ovio5OxAABkL64QSpBDvWpjWKe8ZN6-rjnKec&s=GpvnQCOnZQ1WzpwkGnw56mlXRibqNGz6nzwt25cQkQw&e=

File Protected Password: cvpr18polyrnnpp

After unzipping the project, copy the polyrnn-pp-pytorch/models into the `polyrnn` (pytorch 1.3 project).
```

## To run DEXTR

Run the dextr [notebook](./notebooks/labelbox/4_Dextr_API.ipynb)

You will need to clone the dextr project at the base of this project:

```
cd nuclei
git clone https://github.com/scaelles/DEXTR-PyTorch
cd DEXTR-PyTorch
cd models/
chmod +x download_dextr_model.sh
./download_dextr_model.sh
cd ..
```

# B. Data Preparation for Annotations

1  All digitalized cytospins are recorded in `Cytospin Dataset Register and dataprep procedures.docx`, including:
 - patient ID
 - thumbnail image
 - marked for training or test dataset
 - setting for tiling into 1024 x 1024 size etc

2  Follow the instruction in `notebooks/labelbox/A_project_setup_using_GraphQLclient.ipynb` to setup a project in Labelbox.  The current setting is for point annotation.  But, with minimal changes, it can be used for polygon or rectangle annotations.  (Note: LabelBox introduced a new editor interface.  Also, the GraphQLclient was written for Python 2.7 which is no longer supported.  To set-up a project directly in LabelBox.)  

3  Upload a group of 1024 x 1024 tiles into LabelBox.  To name for dataset using the item number in the Cytospin Dataset Register.

4  Inspect the setting in LabelBox before noticing the assessors.

# C. Interpersonal Variability Test (IVT) for Assessors using MS Excel and R

Preparation: In LabelBox, download labels for a project with .json format.  Move the file from Download folder to `../../data/meta/` and rename it.

1  Run `notebooks/labelbox/B_Interpersonal_Variability_Test_result_from_individual_assessor.ipynb` to save individual IVT result into a .csv file.  Repeat the process for all assessors.

2  Update individual assessor's result to `IVT#_Mean_Difference_Plot.xlxs.`

For ```R``` related analysis, follow https://docs.anaconda.com/anaconda/navigator/tutorials/r-lang/ to open an `R` environment.

3  In `statistic` folder, use `R` kernel to run `R_IVT[#]_Intraclass_Corelation_Coefficient.ipynb` for calculating ICC.  This folder, stored statistic data, files and notebooks.

#  D. For Visual Inspections amongst Assessors

1. Run `notebooks/labelbox/D_Inspect_points_annotations_CS25images.ipynb`

2. Save the results as .html or .pdf for assessors.
