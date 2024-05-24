import streamlit as st
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from nilearn import image as nli
from nilearn import datasets
from nilearn.masking import compute_brain_mask
import pandas as pd
import statsmodels.api as sm

# Load an atlas for segmentation
atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr50-1mm')
atlas_labels = atlas['labels']

def load_nii_file(uploaded_file):
    file_holder = nib.FileHolder(fileobj=uploaded_file)
    nii = nib.Nifti1Image.from_file_map({'header': file_holder, 'image': file_holder})
    return nii

def plot_slice(data, slice_number, axis=0):
    fig, ax = plt.subplots()
    if axis == 0:  # Sagittal
        slice_data = data[slice_number, :, :]
    elif axis == 1:  # Coronal
        slice_data = data[:, slice_number, :]
    else:  # Axial
        slice_data = data[:, :, slice_number]
    ax.imshow(slice_data.T, cmap="gray", origin="lower")
    ax.axis('off')
    return fig

def plot_highlighted_slice(data, slice_number, axis, labels_img, region_label):
    fig, ax = plt.subplots()
    if axis == 0:  # Sagittal
        slice_data = data[slice_number, :, :]
        roi_data = labels_img.get_fdata()[slice_number, :, :] == region_label
    elif axis == 1:  # Coronal
        slice_data = data[:, slice_number, :]
        roi_data = labels_img.get_fdata()[:, slice_number, :] == region_label
    else:  # Axial
        slice_data = data[:, :, slice_number]
        roi_data = labels_img.get_fdata()[:, :, slice_number] == region_label
    ax.imshow(slice_data.T, cmap="gray", origin="lower")
    ax.contour(roi_data, colors='red', linewidths=0.5)
    ax.axis('off')
    return fig

def plot_3d_brain(data, labels_img):
    # Get coordinates where intensity is above 95th percentile
    coords = np.array(np.nonzero(data > np.percentile(data, 95)))
    x, y, z = coords
    intensities = data[x, y, z]
    regions = labels_img.get_fdata()[x, y, z]

    # Create hover texts for each point
    hover_texts = [f"Region: {atlas_labels[int(region)]}<br>Intensity: {intensity:.2f}"
                   for region, intensity in zip(regions, intensities)]

    # Create 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=3,
            color=intensities,
            colorscale='Viridis',
            colorbar=dict(
                title='Proton Density',
                thickness=15,
                xpad=10
            ),
            opacity=0.8
        ),
        text=hover_texts,
        hoverinfo='text'
    )])

    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='Left-Right'),
            yaxis=dict(title='Posterior-Anterior'),
            zaxis=dict(title='Inferior-Superior')
        ),
        margin=dict(r=10, l=10, b=10, t=10)
    )
    return fig

def skull_strip(nii_data):
    brain_mask = compute_brain_mask(nii_data)
    masked_img = nli.math_img("img1 * img2", img1=nii_data, img2=brain_mask)
    return masked_img

def apply_segmentation(nii_data, atlas_data):
    labels_img = nli.resample_to_img(source_img=atlas_data.maps, target_img=nii_data, interpolation='nearest')
    return labels_img

def calculate_region_statistics(data, labels_img):
    regions = np.unique(labels_img.get_fdata())
    stats = []
    for region in regions:
        if region == 0:  # skip background
            continue
        region_voxels = data[labels_img.get_fdata() == region]
        mean_intensity = np.mean(region_voxels)
        volume = np.count_nonzero(region_voxels)
        stats.append({'Region': atlas_labels[int(region)], 'Mean Intensity': mean_intensity, 'Volume': volume})
    return stats

def individual_statistics(data, labels_img, region_label):
    region_data = data[labels_img.get_fdata() == region_label]
    mean_intensity = np.mean(region_data)
    volume = np.count_nonzero(region_data)
    return {'Region': atlas_labels[int(region_label)], 'Mean Intensity': mean_intensity, 'Volume': volume}

def generate_charts(stats_df):
    fig_scatter = px.scatter(stats_df, x='Volume', y='Mean Intensity', color='Region',
                             title="Volume vs Intensity Scatter Plot")
    fig_pie = px.pie(stats_df, values='Volume', names='Region', title="Volume Distribution by Region")
    return fig_scatter, fig_pie

st.title('Brain Analysis')

uploaded_file = st.file_uploader("Choose a NII file", type=["nii", "gz"])

if uploaded_file:
    nii_data = load_nii_file(uploaded_file)
    if nii_data:
        data = nii_data.get_fdata()

        with st.sidebar.expander("2D Scan"):
            axis = st.selectbox('Select the axis for slicing:', options=['Sagittal', 'Coronal', 'Axial'], index=2)
            axis_map = {'Sagittal': 0, 'Coronal': 1, 'Axial': 2}
            slice_num = st.slider('Select Slice Number', min_value=0, max_value=data.shape[axis_map[axis]] - 1,
                                  value=data.shape[axis_map[axis]] // 2)

            skull_strip_option = st.checkbox("Apply Skull Stripping")

            if st.checkbox("Highlight Region"):
                segmentation_applied = st.checkbox("Apply Segmentation for Highlighting")
                if segmentation_applied:
                    if skull_strip_option:
                        nii_data = skull_strip(nii_data)
                        data = nii_data.get_fdata()
                    labels_img = apply_segmentation(nii_data, atlas)
                    region_index = st.selectbox("Select Region to Highlight",
                                                options=[(i, atlas_labels[i]) for i in np.unique(labels_img.get_fdata().astype(int)) if i != 0])
                    region_label = region_index[0]
                    fig = plot_highlighted_slice(data, slice_num, axis_map[axis], labels_img, region_label)
                else:
                    if skull_strip_option:
                        nii_data = skull_strip(nii_data)
                        data = nii_data.get_fdata()
                    fig = plot_slice(data, slice_num, axis=axis_map[axis])
            else:
                if skull_strip_option:
                    nii_data = skull_strip(nii_data)
                    data = nii_data.get_fdata()
                fig = plot_slice(data, slice_num, axis=axis_map[axis])

            st.pyplot(fig)

        with st.sidebar.expander("3D Scan"):
            segmentation_applied = st.checkbox("Apply Segmentation")
            if segmentation_applied:
                if skull_strip_option:
                    nii_data = skull_strip(nii_data)
                    data = nii_data.get_fdata()
                labels_img = apply_segmentation(nii_data, atlas)
                region_index = st.selectbox("Select Region for Statistics",
                                            options=[(i, atlas_labels[i]) for i in np.unique(labels_img.get_fdata().astype(int)) if i != 0])
                region_label = region_index[0]
                region_stats = individual_statistics(data, labels_img, region_label)
                st.write(f"Region: {region_stats['Region']}")
                st.write(f"Mean Intensity: {region_stats['Mean Intensity']:.2f}")
                st.write(f"Volume: {region_stats['Volume']}")

            uploaded_csv = st.file_uploader("Upload time-series data (CSV) for GLM Analysis", type=["csv"])
            if uploaded_csv:
                time_series = pd.read_csv(uploaded_csv)
                time_series = sm.add_constant(time_series)  # Add a constant term for the intercept

                num_time_points = data.shape[-1]  # Number of time points in NIfTI file
                num_time_points = min(num_time_points, time_series.shape[0])  # Use the smaller of the two

                if num_time_points >= 2:
                    selected_region = st.selectbox("Select Region for GLM Analysis",
                                                   [(i, atlas_labels[i]) for i in np.unique(labels_img.get_fdata().astype(int)) if i != 0])
                    region_index = selected_region[0]
                    region_data = data[labels_img.get_fdata() == region_index].flatten()[:num_time_points]
                    time_series = time_series.iloc[:num_time_points]

        with st.expander("3D View"):
            if segmentation_applied:
                fig_3d = plot_3d_brain(data, labels_img)
                st.plotly_chart(fig_3d, use_container_width=True)

        if segmentation_applied:
            stats = calculate_region_statistics(data, labels_img)
            stats_df = pd.DataFrame(stats)

            with st.expander("Statistical Analysis"):
                fig_scatter, fig_pie = generate_charts(stats_df)
                st.plotly_chart(fig_scatter, use_container_width=True)
                st.plotly_chart(fig_pie, use_container_width=True)

            if uploaded_csv and num_time_points >= 2:
                with st.expander("GLM Results"):
                    glm_model = sm.OLS(region_data, time_series)
                    results = glm_model.fit()
                    st.write(results.summary())

                    t_test = results.t_test([1] + [0] * (time_series.shape[1] - 1))
                    st.write("T-test results:")
                    st.write(t_test.summary_frame())

                    st.write("**GLM Analysis Report:**")
                    st.write(f"The General Linear Model (GLM) analysis for the selected region '{selected_region[1]}' revealed the following key results:")
                    st.write(f"- **R-squared**: {results.rsquared:.4f}, indicating that {results.rsquared*100:.2f}% of the variance in the brain activity is explained by the model.")
                    st.write(f"- **F-statistic**: {results.fvalue:.2f} with a p-value of {results.f_pvalue:.4f}, suggesting that the overall model is statistically significant.")
                    st.write(f"- **Coefficients**:")
                    coef_df = pd.DataFrame({"Coefficient": results.params, "Std Error": results.bse, "t-value": results.tvalues, "p-value": results.pvalues})
                    st.table(coef_df)

                    if t_test.tvalue[0] > 1.96 or t_test.tvalue[0] < -1.96:
                        st.write("The t-test for the intercept is statistically significant at the 0.05 level, indicating that there is a significant relationship between the intercept and the brain activity in the selected region.")
                    else:
                        st.write("The t-test for the intercept is not statistically significant at the 0.05 level, suggesting that the relationship between the intercept and the brain activity in the selected region is not significant.")
            else:
                st.write("Time series data not inputted or not enough data points for GLM analysis. Please upload time series data with at least 2 points for GLM analysis.")
