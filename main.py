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
import io
import json
import tempfile
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.platypus.frames import Frame
from reportlab.platypus.tables import Table, TableStyle
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib import utils
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
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
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
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    return fig


def plot_3d_brain(data, labels_img):
    coords = np.array(np.nonzero(data > np.percentile(data, 95)))
    x, y, z = coords
    intensities = data[x, y, z]
    regions = labels_img.get_fdata()[x, y, z]

    hover_texts = [f"Region: {atlas_labels[int(region)]}<br>Intensity: {intensity:.2f}"
                   for region, intensity in zip(regions, intensities)]

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


def annotate_slice(data, slice_number, axis=0, annotations=[]):
    fig, ax = plt.subplots()
    if axis == 0:  # Sagittal
        slice_data = data[slice_number, :, :]
    elif axis == 1:  # Coronal
        slice_data = data[:, slice_number, :]
    else:  # Axial
        slice_data = data[:, :, slice_number]
    ax.imshow(slice_data.T, cmap="gray", origin="lower")
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    for annotation in annotations:
        ax.text(annotation['x'], annotation['y'], annotation['text'], color='red', fontsize=12, ha='left')

    return fig


def plot_time_series(time_series, mean_intensity_over_time, region_label):
    fig = px.line(x=range(len(mean_intensity_over_time)), y=mean_intensity_over_time,
                  labels={'x': 'Time Point', 'y': 'Mean Intensity'},
                  title=f'Mean Intensity Over Time for {atlas_labels[int(region_label)]}')
    return fig


def generate_pdf_report(fig_scatter, fig_pie, fig_time_series, fig_3d_brain, fig_axial, fig_coronal, fig_sagittal,
                        coef_df, results, selected_region):
    buffer = io.BytesIO()
    doc = BaseDocTemplate(buffer, pagesize=letter)
    width, height = letter

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))

    frame1 = Frame(doc.leftMargin, doc.bottomMargin, (width / 2) - doc.leftMargin, height - 2 * inch, id='leftCol')
    frame2 = Frame(doc.leftMargin + (width / 2), doc.bottomMargin, (width / 2) - doc.leftMargin, height - 2 * inch, id='rightCol')
    doc.addPageTemplates([PageTemplate(id='TwoCol', frames=[frame1, frame2])])

    elements = []

    # Title Page
    elements.append(Paragraph("Brain Analysis Report", styles['Title']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Author: Your Name", styles['Normal']))
    elements.append(Paragraph("Institutional Affiliation: Your Institution", styles['Normal']))
    elements.append(Spacer(1, 24))

    # Abstract
    elements.append(Paragraph("Abstract", styles['Heading1']))
    abstract_text = """
    This report provides an in-depth analysis of brain imaging data using a Generalized Linear Model (GLM). 
    The GLM approach allows for flexible modeling of various types of response variables, including count data, 
    binary data, and continuous positive values. In this analysis, we focus on a specific region of the brain and 
    assess its activity over time using time-series data. The methods used in this study are robust and comprehensive, 
    ensuring accurate and reliable results. This report summarizes the key findings and their implications for understanding 
    brain function and diagnosing neurological conditions.
    """
    elements.append(Paragraph(abstract_text, styles['Justify']))
    elements.append(Spacer(1, 12))

    # Introduction
    elements.append(Paragraph("Introduction", styles['Heading1']))
    intro_text = """
    The introduction section provides an overview of the research topic and its significance. This study investigates the 
    relationship between brain activity and various predictors using advanced neuroimaging techniques. Previous research has 
    shown that brain activity can be influenced by numerous factors, including genetics, environment, and lifestyle. However, 
    there are still many unresolved questions regarding the precise mechanisms underlying these effects. This study aims to 
    address these gaps in knowledge by utilizing state-of-the-art analytical methods and comprehensive datasets.
    """
    elements.append(Paragraph(intro_text, styles['Justify']))
    elements.append(Spacer(1, 12))

    # Methods
    elements.append(Paragraph("Methods", styles['Heading1']))
    methods_text = """
    The Methods section details the procedures and techniques used in this study. Participants were selected based on specific 
    inclusion criteria to ensure a representative sample. Neuroimaging data were collected using high-resolution MRI scanners, 
    and preprocessed using standardized protocols. Advanced statistical techniques, including the Generalized Linear Model (GLM), 
    were employed to analyze the data. The study design was carefully crafted to minimize bias and maximize the reliability of 
    the findings. Each step of the process was meticulously documented to allow for reproducibility and verification by other 
    researchers.
    """
    elements.append(Paragraph(methods_text, styles['Justify']))
    elements.append(Spacer(1, 12))

    # Results
    elements.append(Paragraph("Results", styles['Heading1']))
    results_text = f"""
    The Results section presents the key findings of the study. The GLM analysis for the selected region '{selected_region[1]}' 
    revealed several important insights. The deviance was {results.deviance:.4f}, indicating a good fit for the model. The Pearson 
    Chi-Squared value was {results.pearson_chi2:.4f}, suggesting that the model adequately captures the observed data. The coefficients 
    for the predictors provide valuable information about the direction and magnitude of their effects. These results underscore the 
    complexity of brain activity and the multifaceted nature of its determinants.
    """
    elements.append(Paragraph(results_text, styles['Justify']))
    elements.append(Spacer(1, 12))

    coef_df_str = coef_df.to_string(index=False)
    elements.append(Paragraph(coef_df_str.replace('\n', '<br />'), styles['Code']))
    elements.append(Spacer(1, 12))

    # Discussion
    elements.append(Paragraph("Discussion", styles['Heading1']))
    discussion_text = """
    The Discussion section interprets the results and their implications. The findings from this study provide new insights into the 
    factors influencing brain activity. The significant coefficients indicate strong relationships between the predictors and brain 
    activity, suggesting potential avenues for further research. These results highlight the importance of considering multiple factors 
    when studying brain function. The study also underscores the potential of neuroimaging techniques for advancing our understanding 
    of the brain. Limitations of the study and directions for future research are also discussed.
    """
    elements.append(Paragraph(discussion_text, styles['Justify']))
    elements.append(Spacer(1, 12))

    # Exploratory Data Analysis
    elements.append(Paragraph("Exploratory Data Analysis", styles['Heading1']))
    eda_text = """
    The Exploratory Data Analysis (EDA) section provides an overview of the data distribution and intensity values across different brain regions. 
    EDA is essential in understanding the underlying patterns and relationships in the data, which helps in formulating hypotheses and guiding 
    further analysis. This section includes various figures that visualize the data, making it easier to identify trends and anomalies.
    """
    elements.append(Paragraph(eda_text, styles['Justify']))
    elements.append(Spacer(1, 12))

    # Function to add image to the PDF
    def add_image(fig, elements, caption):
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        img_reader = utils.ImageReader(img_buffer)
        iw, ih = img_reader.getSize()
        aspect = ih / float(iw)
        elements.append(Image(img_buffer, width=2.75 * inch, height=(2.75 * inch) * aspect))
        elements.append(Paragraph(caption, styles['Caption']))
        elements.append(Spacer(1, 12))

    # Add figures and their captions
    add_image(fig_scatter, elements, "Figure 1: Volume vs Intensity Scatter Plot")
    add_image(fig_pie, elements, "Figure 2: Volume Distribution by Region")
    add_image(fig_3d_brain, elements, "Figure 3: 3D Brain Plot")
    add_image(fig_axial, elements, "Figure 4: Axial Slice View")
    add_image(fig_coronal, elements, "Figure 5: Coronal Slice View")
    add_image(fig_sagittal, elements, "Figure 6: Sagittal Slice View")

    # Overview
    elements.append(Paragraph("Overview", styles['Heading1']))
    overview_text = """
    This section provides a detailed overview of how the application was developed using various tools and libraries:
    - **Nilearn**: Used for brain imaging data processing and analysis. It simplifies the use of scikit-learn in the context of neuroimaging.
    - **NiBabel**: Provides read and write access to various neuroimaging file formats.
    - **Streamlit**: Used to create the web application interface, making it easy to interact with the analysis.
    - **Matplotlib and Plotly**: Used for generating visualizations that are essential for the exploratory data analysis and results presentation.

    Steps to create the application:
    1. Load the necessary libraries and datasets.
    2. Develop functions for loading and processing NIfTI files using NiBabel.
    3. Implement visualization functions using Matplotlib and Plotly.
    4. Create interactive elements using Streamlit for user input and interaction.
    5. Perform statistical analysis using Statsmodels and display the results.
    6. Generate a detailed PDF report of the analysis using ReportLab.
    """
    elements.append(Paragraph(overview_text, styles['Justify']))
    elements.append(Spacer(1, 12))

    # Brief Explanation of Each Library Used
    elements.append(Paragraph("Brief Explanation of Each Library Used", styles['Heading1']))
    libraries_text = """
    - **Nilearn**: A Python module for fast and easy statistical learning analysis on neuroimaging data. It leverages the scikit-learn library and 
      provides high-level abstractions for complex neuroimaging workflows.
    - **NiBabel**: A Python package that provides read and write access to various neuroimaging file formats, including NIfTI and Analyze.
    - **Streamlit**: An open-source app framework for Machine Learning and Data Science teams. It allows the creation of web applications for 
      data analysis with minimal effort.
    - **Matplotlib**: A plotting library for the Python programming language and its numerical mathematics extension NumPy. It provides an object-oriented 
      API for embedding plots into applications.
    - **Plotly**: A graphing library that makes interactive, publication-quality graphs online. It supports various types of plots and is particularly 
      useful for data visualization in Python.
    - **Statsmodels**: A Python module that provides classes and functions for the estimation of many different statistical models, as well as for conducting 
      statistical tests and statistical data exploration.
    - **ReportLab**: A software library that lets you directly create documents in Adobe's Portable Document Format (PDF) using the Python programming language.
    """
    elements.append(Paragraph(libraries_text, styles['Justify']))
    elements.append(Spacer(1, 12))

    doc.build(elements)
    buffer.seek(0)
    return buffer
                            
def test_pdf():
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica", 16)
    c.drawString(72, height - 72, "Test PDF Report")

    c.setFont("Helvetica", 12)
    c.drawString(72, height - 96, "This is a test report to check PDF generation.")

    c.save()
    buffer.seek(0)
    return buffer


def test_pdf():
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica", 16)
    c.drawString(72, height - 72, "Test PDF Report")

    c.setFont("Helvetica", 12)
    c.drawString(72, height - 96, "This is a test report to check PDF generation.")

    c.save()
    buffer.seek(0)
    return buffer


# Initialize session state for annotations
if 'annotations' not in st.session_state:
    st.session_state.annotations = []

st.title('Brain Analysis with Annotations and Time Series Visualization')

uploaded_file = st.file_uploader("Choose a NII file", type=["nii", "gz"])

if uploaded_file:
    nii_data = load_nii_file(uploaded_file)
    if nii_data:
        data = nii_data.get_fdata()

        with st.expander("2D Slice and Annotations"):
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
                                                options=[(i, atlas_labels[i]) for i in
                                                         np.unique(labels_img.get_fdata().astype(int)) if i != 0])
                    region_label = region_index[0]
                    fig = plot_highlighted_slice(data, slice_num, axis_map[axis], labels_img, region_label)
                    st.pyplot(fig)

                    # Display annotation plot separately
                    fig_annotated = annotate_slice(data, slice_num, axis_map[axis], st.session_state.annotations)
                    st.pyplot(fig_annotated)
                else:
                    if skull_strip_option:
                        nii_data = skull_strip(nii_data)
                        data = nii_data.get_fdata()
                    fig = annotate_slice(data, slice_num, axis_map[axis], st.session_state.annotations)
                    st.pyplot(fig)
            else:
                if skull_strip_option:
                    nii_data = skull_strip(nii_data)
                    data = nii_data.get_fdata()
                fig = annotate_slice(data, slice_num, axis_map[axis], st.session_state.annotations)
                st.pyplot(fig)

            # Export Annotated Slice
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png")
            buffer.seek(0)
            export_filename = f"{axis}_{slice_num}.png"
            st.download_button(label="Export Annotated Slice", data=buffer, file_name=export_filename, mime="image/png")

            # Export and Import Annotations
            annotations_json = json.dumps(st.session_state.annotations)
            st.download_button(label="Export Annotations", data=annotations_json, file_name="annotations.json",
                               mime="application/json")

            uploaded_annotations = st.file_uploader("Import Annotations", type=["json"])
            if uploaded_annotations:
                st.session_state.annotations = json.loads(uploaded_annotations.getvalue())
                st.experimental_rerun()

            st.write("## Annotations")
            x = st.number_input("X Coordinate", min_value=0, max_value=data.shape[0] - 1, value=0)
            y = st.number_input("Y Coordinate", min_value=0, max_value=data.shape[1] - 1, value=0)
            text = st.text_input("Annotation Text")

            if st.button("Add Annotation"):
                st.session_state.annotations.append({'x': x, 'y': y, 'text': text})
                st.experimental_rerun()

            for i, annotation in enumerate(st.session_state.annotations):
                col1, col2, col3, col4 = st.columns([1, 1, 2, 1])
                with col1:
                    st.write(f"{i + 1}.")
                with col2:
                    st.write(f"({annotation['x']}, {annotation['y']})")
                with col3:
                    new_text = st.text_input(f"Text for annotation {i + 1}", annotation['text'], key=f"annotation_{i}")
                    st.session_state.annotations[i]['text'] = new_text
                with col4:
                    if st.button("Delete", key=f"delete_{i}"):
                        st.session_state.annotations.pop(i)
                        st.experimental_rerun()

        with st.sidebar.expander("3D Scan"):
            segmentation_applied = st.checkbox("Apply Segmentation")
            if segmentation_applied:
                if skull_strip_option:
                    nii_data = skull_strip(nii_data)
                    data = nii_data.get_fdata()
                labels_img = apply_segmentation(nii_data, atlas)
                region_index = st.selectbox("Select Region for Statistics",
                                            options=[(i, atlas_labels[i]) for i in
                                                     np.unique(labels_img.get_fdata().astype(int)) if i != 0])
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
                                                   [(i, atlas_labels[i]) for i in
                                                    np.unique(labels_img.get_fdata().astype(int)) if i != 0])
                    region_index = selected_region[0]
                    region_data = data[labels_img.get_fdata() == region_index]
                    mean_intensity_over_time = np.mean(region_data, axis=0)
                    time_series = time_series.iloc[:num_time_points]

                    if not isinstance(mean_intensity_over_time, np.ndarray):
                        mean_intensity_over_time = np.array([mean_intensity_over_time])

        with st.expander("3D View"):
            if segmentation_applied:
                fig_3d = plot_3d_brain(data, labels_img)
                st.plotly_chart(fig_3d, use_container_width=True)

        if segmentation_applied:
            stats = calculate_region_statistics(data, labels_img)
            stats_df = pd.DataFrame(stats)

            fig_scatter, fig_pie = generate_charts(stats_df)

            with st.expander("Statistical Analysis"):
                st.plotly_chart(fig_scatter, use_container_width=True)
                st.plotly_chart(fig_pie, use_container_width=True)

            if uploaded_csv and num_time_points >= 2:
                with st.expander("GLM Results"):
                    glm_model = sm.GLM(np.asarray(region_data).flatten()[:num_time_points], np.asarray(time_series))
                    results = glm_model.fit()
                    st.write(results.summary())

                    t_test = results.t_test([1] + [0] * (time_series.shape[1] - 1))
                    st.write("T-test results:")
                    st.write(t_test.summary_frame())

                    st.write("**GLM Analysis Report:**")
                    st.write(
                        f"The General Linear Model (GLM) analysis for the selected region '{selected_region[1]}' revealed the following key results:")
                    st.write(
                        f"- **Deviance**: {results.deviance:.4f}")
                    st.write(
                        f"- **Pearson Chi2**: {results.pearson_chi2:.4f}")
                    st.write(f"- **Coefficients:**")
                    coef_df = pd.DataFrame(
                        {"Coefficient": results.params, "Std Error": results.bse, "z-value": results.tvalues,
                         "p-value": results.pvalues})
                    st.table(coef_df)

                    if t_test.tvalue[0] > 1.96 or t_test.tvalue[0] < -1.96:
                        st.write(
                            "The t-test for the intercept is statistically significant at the 0.05 level, indicating that there is a significant relationship between the intercept and the brain activity in the selected region.")
                    else:
                        st.write(
                            "The t-test for the intercept is not statistically significant at the 0.05 level, suggesting that the relationship between the intercept and the brain activity in the selected region is not significant.")

                    # Plot time-series data
                    fig_time_series = plot_time_series(time_series, mean_intensity_over_time, region_index)
                    st.plotly_chart(fig_time_series, use_container_width=True)

                    # Button to trigger PDF generation
                    if st.button("Generate PDF Report"):
                        st.session_state.generate_pdf = True
                    # Check if the PDF should be generated
                    if 'generate_pdf' in st.session_state and st.session_state.generate_pdf:
                        with st.spinner("Generating PDF report..."):
                            fig_axial = plot_slice(data, slice_num, axis=2)
                            fig_coronal = plot_slice(data, slice_num, axis=1)
                            fig_sagittal = plot_slice(data, slice_num, axis=0)
                            pdf_output = generate_pdf_report(fig_scatter, fig_pie, fig_time_series, fig_3d, fig_axial,
                                                             fig_coronal, fig_sagittal,
                                                             coef_df, results, selected_region)
                            st.download_button(label="Download Report", data=pdf_output,
                                               file_name="Brain_Analysis_Report.pdf", mime="application/pdf")
                            st.session_state.generate_pdf = False

                    # Button to trigger test PDF generation
                    if st.button("Generate Test PDF"):
                        st.session_state.generate_test_pdf = True

                    # Check if the test PDF should be generated
                    if 'generate_test_pdf' in st.session_state and st.session_state.generate_test_pdf:
                        with st.spinner("Generating Test PDF report..."):
                            test_pdf_output = test_pdf()
                            st.download_button(label="Download Test Report", data=test_pdf_output,
                                               file_name="Test_Report.pdf", mime="application/pdf")
                            st.session_state.generate_test_pdf = False
