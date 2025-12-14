<p align="center">
  <img src="data/census.jpg" alt="Census Logo" width="200"/>
</p>

# "Census Analysis"

This repo contains the code, results, models, report for Census Analysis.

- There are two major parts in this project:
1. Supervised Learning (Classification)
2. Unsupervised Learning (Clustering)

### Datasets used
---
- The data set contains weighted census data extracted from the 1994 and 1995 Current Population Surveys conducted by the U.S. Census Bureau. Each line of the data set (censusbureau.data) contains 40 demographic and employment related variables as well as a weight for the observation and a label for each observation, which indicates whether a particular population component had an income that is greater than or less than $50k.

### Project Objective
---
- Develop a classification model to predict and classify the data into two groups: People who earn an income less than $50,000 and those who earn more than $50,000
- Create the segmentation model and demonstrate how the resulting groups differ from one another.

### Files Description
---
- ```data```: contains the data files including columns, the original data file and refined excel file.
- ```experiments```: contains the experiment notebooks for classification and segmentation.
- ```models```: contains the trained models for classification and segmentation.
- ```1_EDA.ipynb```: contains exploratory data analysis on the census data.
- ```2_classification.ipynb```: contains code for data preprocessing, train-test split, hyperparameter tuning, training and evaluation of classification model.
- ```3_segmentation.ipynb```: contains code for data preprocessing, dimensionality reduction, clustering algorithm, and interpretation of clusters.
- ```environment.yml```: libraries required to run the notebooks are defined in this file.
- ```ML-TakehomeProject.pdf```: Instruction file
- ```Report.pdf```: Client Report file

### How to Run
---
- This project uses Python and other data science libraries. This project was developed using Google Colab Pro with a High RAM CPU. It is recommended that you download or clone the repo and directly run files on Google Colab. 
- If you wish to run the files on local, here are the steps to do so:
1. Install Conda:  
If you do not already have it install, download Miniconda for the reqiured OS [Miniconda Installation Guide](https://www.anaconda.com/docs/getting-started/miniconda/main#latest-miniconda-installer-links)
2. Create the conda environment
Once the conda is installed, you can create a dedicated environment for this project to avoid conflicts. Now, navigate to project's root directory and in your terminal run the following commands:
```bash
conda env create -f environment.yml
conda activate census_analysis
```
3. Once the environment setup is done, and environment is activated, you can run the three files with Jupyter Notebook. In the terminal, run the following command:
```bash
jupyter notebook
```
4. Running this command will automatically oopen a new tab in your web browser, showing a Jupyter Notebook dashboard. Ensure the correct python kernel is selected.
5. Run these files: ```1_EDA.ipynb```, ```2_classification.ipynb```, ```3_segmentation.ipynb```
6. **Note** that these file will require at lease at HIGH RAM CPU to run, as the Hyperparameter Configurations will also run, it will take 2 to 3 hours to run the classification file.