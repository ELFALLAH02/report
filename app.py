import streamlit as st
import pandas as pd
import os
from components.dashboard import render_dashboard
from components.summary import render_summary
from components.model_ranking import render_model_ranking
from components.model_performance import render_model_performance
from components.detailed_performance import render_detailed_performance
from components.performance_by_year import render_performance_by_year
from components.precision_trends import render_precision_trends
from components.recall_trends import render_recall_trends
from components.precision_distribution import render_precision_distribution
from components.recall_distribution import render_recall_distribution
from components.tp_fp_fn import render_tp_fp_fn
from components.performance_by_context import render_performance_by_context
from components.error_analysis import render_error_analysis
from components.correlation_heatmap import render_correlation_heatmap
from components.top_images import render_top_images
from components.export_data import render_export_data
from components.conclusion import render_conclusion
from components.help_section import render_help_section
from utils.utils import load_data, calculate_model_metrics

# Set page configuration
st.set_page_config(page_title="Advanced Model Evaluation Report", layout="wide")

def main():
    st.title("Advanced Model Evaluation Report")
    st.markdown("Evaluate machine learning models for object detection in citrus groves. Use filters to explore performance metrics and visualizations.")

    # Data folder path
    data_folder = r"data"
    if not os.path.exists(data_folder):
        st.error(f"Data folder not found: {data_folder}")
        return

    # Load data
    with st.spinner("Loading data..."):
        df, model_nums = load_data(data_folder)
    if df is None:
        return

    # Check for required columns
    required_cols = ['year', 'domaine', 'porte_greffe', 'parcelle', 'filename']
    for mn in model_nums:
        required_cols.extend([f'precision_{mn}', f'recall_{mn}', f'tp_{mn}', f'fp_{mn}', f'fn_{mn}'])
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
        return

    # Filters
    with st.expander("Filters", expanded=True):
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            models = [f'Model {i}' for i in model_nums]
            selected_model = st.selectbox("Select Model", ['All Models'] + models, key='model', help="Choose a specific model or view all models.")
        with col2:
            years = sorted(df['year'].unique().astype(str))
            selected_year = st.selectbox("Select Year", ['All Years'] + years, key='year', help="Filter by year of data collection.")
        with col3:
            domaines = sorted(df['domaine'].unique())
            selected_domaine = st.selectbox("Select Domaine", ['All Domaines'] + domaines, key='domaine', help="Filter by domaine (e.g., vineyard or region).")
        with col4:
            porte_greffes = sorted(df['porte_greffe'].unique())
            selected_porte_greffe = st.selectbox("Select Porte Greffe", ['All Porte Greffes'] + porte_greffes, key='porte_greffe', help="Filter by rootstock type.")
        with col5:
            parcelles = sorted(df['parcelle'].unique().astype(str))
            selected_parcelle = st.selectbox("Select Parcelle", ['All Parcelles'] + parcelles, key='parcelle', help="Filter by specific plot or parcel.")

    # Filter data
    filtered_df = df.copy()
    if selected_year != 'All Years':
        filtered_df = filtered_df[filtered_df['year'].astype(str) == selected_year]
    if selected_domaine != 'All Domaines':
        filtered_df = filtered_df[filtered_df['domaine'] == selected_domaine]
    if selected_porte_greffe != 'All Porte Greffes':
        filtered_df = filtered_df[filtered_df['porte_greffe'] == selected_porte_greffe]
    if selected_parcelle != 'All Parcelles':
        filtered_df = filtered_df[filtered_df['parcelle'].astype(str) == selected_parcelle]

    # Calculate model metrics
    model_metrics = calculate_model_metrics(filtered_df, model_nums)
    metrics_df = pd.DataFrame(model_metrics)

    # Render components
    render_dashboard(metrics_df, model_nums, filtered_df)
    render_summary(metrics_df, years)
    render_model_ranking(metrics_df)
    render_model_performance(metrics_df)
    render_detailed_performance(selected_model, model_metrics, metrics_df)
    render_performance_by_year(filtered_df, model_nums)
    render_precision_trends(filtered_df, model_nums)
    render_recall_trends(filtered_df, model_nums)
    render_precision_distribution(filtered_df, model_nums)
    render_recall_distribution(filtered_df, model_nums)
    render_tp_fp_fn(filtered_df, model_nums)
    render_performance_by_context(filtered_df, model_nums)
    render_error_analysis(filtered_df, model_nums)
    render_correlation_heatmap(filtered_df, model_nums)
    render_top_images(filtered_df, model_nums)
    render_export_data(filtered_df)
    render_conclusion(metrics_df)
    render_help_section()

if __name__ == "__main__":
    main()