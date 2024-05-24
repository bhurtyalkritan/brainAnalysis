# Brain Analysis

This is a Streamlit application for analyzing brain NIfTI (Neuroimaging Informatics Technology Initiative) files. The application provides various features for visualizing and analyzing brain scans, including:

## Features

1. **2D Slice Visualization**
  - View brain slices along the sagittal, coronal, and axial axes.
  - Apply skull stripping to remove non-brain voxels.
  - Highlight specific brain regions using an atlas.

2. **3D Brain Visualization**
  - Generate a 3D scatter plot of voxels with intensities above a specified percentile.
  - Interactive hover tooltips displaying the brain region and intensity value.

3. **Statistical Analysis**
  - Calculate mean intensity and volume for each brain region.
  - Generate scatter plots and pie charts for visualizing region statistics.

4. **General Linear Model (GLM) Analysis**
  - Upload time-series data (CSV) for GLM analysis.
  - Select a brain region for GLM analysis.
  - Perform GLM analysis and display the results, including coefficients, t-tests, and summary statistics.

## Usage

1. Upload a NIfTI file using the file uploader.
2. Explore the various features and options in the sidebar.
3. Visualize brain slices and apply desired options (e.g., skull stripping, region highlighting, segmentation).
4. View the 3D brain visualization and hover over voxels for region and intensity information.
5. Analyze region statistics and generate charts (scatter plot, pie chart).
6. (Optional) Upload time-series data (CSV) and select a brain region for GLM analysis.
7. Inspect the GLM analysis results, including coefficients, t-tests, and summary statistics.

## Requirements

- Python 3.x
- Streamlit
- nibabel
- NumPy
- Matplotlib
- Plotly
- Nilearn
- Pandas
- Statsmodels

## Installation

1. Clone the repository or download the source code.
2. Install the required dependencies by running `pip install -r requirements.txt`.
3. Run the Streamlit application by executing `streamlit run app.py`.
