import streamlit as st
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from nilearn import image as nli
from nilearn import datasets
from nilearn.plotting import plot_roi, show
import pandas as pd
import statsmodels.api as sm


atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr50-1mm')


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


def plot_3d_brain(data):
    coords = np.array(np.nonzero(data > np.percentile(data, 95)))
    x, y, z = coords
    intensity = data[x, y, z]
    fig = go.Figure(data=[go.Mesh3d(
        x=x, y=y, z=z,
        intensity=intensity,
        colorscale='Viridis',
        opacity=0.5,
        showscale=True
    )])
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        margin=dict(r=10, l=10, b=10, t=10)
    )
    return fig


def skull_strip(nii_data):
    masked_img = nli.math_img("np.where(img > np.percentile(img, 60), img, 0)", img=nii_data)
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
        stats.append({'Region': atlas.labels[int(region)], 'Mean Intensity': mean_intensity, 'Volume': volume})
    return stats


def individual_statistics(data, labels_img, region_label):
    region_data = data[labels_img.get_fdata() == region_label]
    mean_intensity = np.mean(region_data)
    volume = np.count_nonzero(region_data)
    return {'Region': atlas.labels[int(region_label)], 'Mean Intensity': mean_intensity, 'Volume': volume}


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

            if st.checkbox("Highlight Region"):
                segmentation_applied = st.checkbox("Apply Segmentation for Highlighting")
                if segmentation_applied:
                    labels_img = apply_segmentation(nii_data, atlas)
                    region_index = st.selectbox("Select Region to Highlight",
                                                options=[(i, label) for i, label in enumerate(atlas.labels) if
                                                         label != 'Background'])
                    region_label = region_index[0]
                    fig = plot_highlighted_slice(data, slice_num, axis_map[axis], labels_img, region_label)
                else:
                    fig = plot_slice(data, slice_num, axis=axis_map[axis])
            else:
                fig = plot_slice(data, slice_num, axis=axis_map[axis])

            st.pyplot(fig)

        with st.sidebar.expander("3D Scan"):
            segmentation_applied = st.checkbox("Apply Segmentation")
            if segmentation_applied:
                labels_img = apply_segmentation(nii_data, atlas)
                region_index = st.selectbox("Select Region for Statistics",
                                            options=[(i, label) for i, label in enumerate(atlas.labels) if
                                                     label != 'Background'])
                region_stats = individual_statistics(data, labels_img, region_index[0])
                st.write(f"Region: {region_stats['Region']}")
                st.write(f"Mean Intensity: {region_stats['Mean Intensity']:.2f}")
                st.write(f"Volume: {region_stats['Volume']}")

            uploaded_csv = st.file_uploader("Upload time-series data (CSV) for GLM Analysis", type=["csv"])
            if uploaded_csv:
                time_series = pd.read_csv(uploaded_csv)
                time_series = sm.add_constant(time_series)

                num_time_points = data.shape[-1] 
                num_time_points = min(num_time_points, time_series.shape[0])  

                if num_time_points >= 2:
                    selected_region = st.selectbox("Select Region for GLM Analysis",
                                                   [label for label in atlas.labels if label != 'Background'])
                    region_index = atlas.labels.index(selected_region)
                    region_data = data[labels_img.get_fdata() == region_index].flatten()[:num_time_points]
                    time_series = time_series.iloc[:num_time_points]

        with st.expander("3D View"):
            fig_3d = plot_3d_brain(data)
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
                    st.write(f"The General Linear Model (GLM) analysis for the selected region '{selected_region}' revealed the following key results:")
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
                st.write(
                    "Time series data not inputted or not enough data points for GLM analysis. Please upload time series data with at least 2 points for GLM analysis.")
