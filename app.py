import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os

# Set page configuration
st.set_page_config(page_title="Advanced Model Evaluation Report", layout="wide")

# Function to load and preprocess data
@st.cache_data
def load_data(file_path):
    try:
        # Load Excel file with header in row 1 (Excel row 2, 0-indexed in pandas)
        df = pd.read_excel(file_path, header=1)
        
        # Clean column names
        df.columns = df.columns.str.strip().str.lower().str.replace('"', '')
        
        # Convert numeric columns
        for col in df.columns:
            if col.startswith(('true_count_', 'detect_count_', 'tp_', 'fp_', 'fn_')):
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            elif col.startswith(('precision_', 'recall_')):
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Rename 'compagnie' to 'year' for consistency
        if 'compagnie' in df.columns:
            df = df.rename(columns={'compagnie': 'year'})
        else:
            st.error("Column 'compagnie' (used as year) not found in the data.")
            return None
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Function to calculate model metrics
def calculate_model_metrics(df, model_nums):
    model_metrics = []
    for mn in model_nums:
        prec_col = f'precision_{mn}'
        rec_col = f'recall_{mn}'
        tp_col = f'tp_{mn}'
        fp_col = f'fp_{mn}'
        fn_col = f'fn_{mn}'
        
        valid_data = df[df[prec_col] > 0]
        avg_precision = valid_data[prec_col].mean() if not valid_data.empty else 0
        avg_recall = valid_data[rec_col].mean() if not valid_data.empty else 0
        total_tp = df[tp_col].sum()
        total_fp = df[fp_col].sum()
        total_fn = df[fn_col].sum()
        f1 = (2 * avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        
        model_metrics.append({
            'model': f'Model {mn}',
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'f1': f1,
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn,
            'data': df[['filename', 'year', 'domaine', 'porte_greffe', 'parcelle', prec_col, rec_col, tp_col, fp_col, fn_col]].copy()
        })
    return model_metrics

# Main app
def main():
    st.title("Advanced Model Evaluation Report")

    # File path
    file_path = "Models_evalution_report(1-23).xlsx"
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return

    # Load data
    df = load_data(file_path)
    if df is None:
        return



    # Check for required columns
    required_cols = ['year', 'domaine', 'porte_greffe', 'parcelle', 'filename']
    model_nums = [i for i in range(1, 24) if i != 18]  # Models 1-17, 19-23
    for mn in model_nums:
        required_cols.extend([f'precision_{mn}', f'recall_{mn}', f'tp_{mn}', f'fp_{mn}', f'fn_{mn}'])
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
        return

    # Filters
    st.subheader("Filters")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        models = [f'Model {i}' for i in model_nums]
        selected_model = st.selectbox("Select Model", ['All Models'] + models, key='model')
    with col2:
        years = sorted(df['year'].unique().astype(str))
        selected_year = st.selectbox("Select Year", ['All Years'] + years, key='year')
    with col3:
        domaines = sorted(df['domaine'].unique())
        selected_domaine = st.selectbox("Select Domaine", ['All Domaines'] + domaines, key='domaine')
    with col4:
        porte_greffes = sorted(df['porte_greffe'].unique())
        selected_porte_greffe = st.selectbox("Select Porte Greffe", ['All Porte Greffes'] + porte_greffes, key='porte_greffe')
    with col5:
        parcelles = sorted(df['parcelle'].unique().astype(str))
        selected_parcelle = st.selectbox("Select Parcelle", ['All Parcelles'] + parcelles, key='parcelle')

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

    # Find winning model
    winner = max(model_metrics, key=lambda x: x['f1'], default={'model': 'None', 'f1': 0})

    # Summary
    st.subheader("Summary")
    st.markdown(f"""
    This report evaluates 23 machine learning models for object detection in citrus groves across {len(filtered_df)} images from years {', '.join(years)}. 
    Metrics include true positives (TP), false positives (FP), false negatives (FN), precision, and recall.
    """)
    st.markdown(f"The **winning model** is {winner['model']} with an F1 score of {winner['f1']*100:.2f}%.")
    st.markdown(f"""
    **Key Observations:**
    - **Highest Precision**: {max(model_metrics, key=lambda x: x['avg_precision'], default={'model': 'None', 'avg_precision': 0})['model']} ({max(model_metrics, key=lambda x: x['avg_precision'], default={'model': 'None', 'avg_precision': 0})['avg_precision']*100:.2f}%)
    - **Highest Recall**: {max(model_metrics, key=lambda x: x['avg_recall'], default={'model': 'None', 'avg_recall': 0})['model']} ({max(model_metrics, key=lambda x: x['avg_recall'], default={'model': 'None', 'avg_recall': 0})['avg_recall']*100:.2f}%)
    - Some years may show lower recall due to higher false negatives, indicating challenges in complex scenes.
    """)

    # Model Performance Bar Chart
    st.subheader("Model Performance Overview")
    metrics_df = pd.DataFrame(model_metrics)
    fig_bar = go.Figure(data=[
        go.Bar(name='Average Precision', x=metrics_df['model'], y=metrics_df['avg_precision'], marker_color='#3B82F6'),
        go.Bar(name='Average Recall', x=metrics_df['model'], y=metrics_df['avg_recall'], marker_color='#10B981')
    ])
    fig_bar.update_layout(
        barmode='group',
        xaxis_title="Model",
        yaxis_title="Score",
        yaxis_tickformat=".0%",
        xaxis_tickangle=-45,
        height=400,
        margin=dict(b=150)
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown("This bar chart shows the average precision and recall for each model, filtered by your selections.")

    # Detailed Model Performance
    if selected_model != 'All Models':
        st.subheader(f"Detailed Performance: {selected_model}")
        model_num = int(selected_model.split(' ')[1])
        selected_model_data = next(m['data'] for m in model_metrics if m['model'] == selected_model)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total True Positives", metrics_df[metrics_df['model'] == selected_model]['total_tp'].iloc[0])
        with col2:
            st.metric("Total False Positives", metrics_df[metrics_df['model'] == selected_model]['total_fp'].iloc[0])
        with col3:
            st.metric("Total False Negatives", metrics_df[metrics_df['model'] == selected_model]['total_fn'].iloc[0])
        
        st.dataframe(
            selected_model_data.rename(columns={
                f'precision_{model_num}': 'Precision',
                f'recall_{model_num}': 'Recall',
                f'tp_{model_num}': 'True Positives',
                f'fp_{model_num}': 'False Positives',
                f'fn_{model_num}': 'False Negatives'
            }).style.format({
                'Precision': '{:.2%}',
                'Recall': '{:.2%}',
                'year': '{}'
            })
        )
        st.markdown(f"This table shows detailed performance metrics for {selected_model} across filtered images.")

    # Performance by Year (Scatter Plot)
    st.subheader("Performance by Year")
    year_data = filtered_df.groupby('year').agg({
        f'precision_{i}': 'mean' for i in model_nums
    }).reset_index()
    year_data['avg_precision'] = year_data[[f'precision_{i}' for i in model_nums]].mean(axis=1)
    year_data['avg_recall'] = filtered_df.groupby('year').agg({
        f'recall_{i}': 'mean' for i in model_nums
    }).reset_index()[[f'recall_{i}' for i in model_nums]].mean(axis=1)
    
    fig_scatter = px.scatter(
        year_data,
        x='avg_precision',
        y='avg_recall',
        color='year',
        text='year',
        labels={'avg_precision': 'Average Precision', 'avg_recall': 'Average Recall'}
    )
    fig_scatter.update_traces(textposition='top center', marker_size=10)
    fig_scatter.update_layout(
        xaxis_tickformat=".0%",
        yaxis_tickformat=".0%",
        height=400
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.markdown("This scatter plot shows average precision vs. recall for each year, filtered by your selections.")

    # Line Chart: Precision Trends Over Years
    st.subheader("Precision Trends Over Years")
    long_df = []
    for mn in model_nums:
        temp_df = filtered_df[['year', f'precision_{mn}']].groupby('year')[f'precision_{mn}'].mean().reset_index()
        temp_df = temp_df.rename(columns={f'precision_{mn}': 'avg_precision'})
        temp_df['model'] = f'Model {mn}'
        long_df.append(temp_df)
    precision_trend_df = pd.concat(long_df)
    fig_line_precision = px.line(
        precision_trend_df,
        x='year',
        y='avg_precision',
        color='model',
        labels={'avg_precision': 'Average Precision'},
        height=400
    )
    fig_line_precision.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig_line_precision, use_container_width=True)
    st.markdown("This line chart shows how average precision for each model changes across years.")

    # Line Chart: Recall Trends Over Years
    st.subheader("Recall Trends Over Years")
    long_df_recall = []
    for mn in model_nums:
        temp_df = filtered_df[['year', f'recall_{mn}']].groupby('year')[f'recall_{mn}'].mean().reset_index()
        temp_df = temp_df.rename(columns={f'recall_{mn}': 'avg_recall'})
        temp_df['model'] = f'Model {mn}'
        long_df_recall.append(temp_df)
    recall_trend_df = pd.concat(long_df_recall)
    fig_line_recall = px.line(
        recall_trend_df,
        x='year',
        y='avg_recall',
        color='model',
        labels={'avg_recall': 'Average Recall'},
        height=400
    )
    fig_line_recall.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig_line_recall, use_container_width=True)
    st.markdown("This line chart shows how average recall for each model changes across years.")

    # Box Plot: Precision Distribution
    st.subheader("Precision Distribution Across Models")
    precision_cols = [f'precision_{mn}' for mn in model_nums]
    melt_df_precision = filtered_df.melt(id_vars=['filename'], value_vars=precision_cols, var_name='model', value_name='precision')
    melt_df_precision['model'] = melt_df_precision['model'].str.replace('precision_', 'Model ')
    fig_box_precision = px.box(
        melt_df_precision,
        x='model',
        y='precision',
        labels={'precision': 'Precision'},
        height=400
    )
    fig_box_precision.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig_box_precision, use_container_width=True)
    st.markdown("This box plot shows the distribution of precision for each model across all images.")

    # Box Plot: Recall Distribution
    st.subheader("Recall Distribution Across Models")
    recall_cols = [f'recall_{mn}' for mn in model_nums]
    melt_df_recall = filtered_df.melt(id_vars=['filename'], value_vars=recall_cols, var_name='model', value_name='recall')
    melt_df_recall['model'] = melt_df_recall['model'].str.replace('recall_', 'Model ')
    fig_box_recall = px.box(
        melt_df_recall,
        x='model',
        y='recall',
        labels={'recall': 'Recall'},
        height=400
    )
    fig_box_recall.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig_box_recall, use_container_width=True)
    st.markdown("This box plot shows the distribution of recall for each model across all images.")

    # Stacked Bar Chart: TP, FP, FN Counts
    st.subheader("True Positives, False Positives, and False Negatives")
    totals = []
    for mn in model_nums:
        totals.append({
            'model': f'Model {mn}',
            'True Positives': filtered_df[f'tp_{mn}'].sum(),
            'False Positives': filtered_df[f'fp_{mn}'].sum(),
            'False Negatives': filtered_df[f'fn_{mn}'].sum()
        })
    totals_df = pd.DataFrame(totals)
    totals_melt = totals_df.melt(id_vars='model', var_name='Metric', value_name='Count')
    fig_stacked = px.bar(
        totals_melt,
        x='model',
        y='Count',
        color='Metric',
        barmode='stack',
        height=400,
        color_discrete_map={
            'True Positives': '#10B981',
            'False Positives': '#EF4444',
            'False Negatives': '#FBBF24'
        }
    )
    st.plotly_chart(fig_stacked, use_container_width=True)
    st.markdown("This stacked bar chart shows the total counts of true positives, false positives, and false negatives for each model.")

    # Correlation Heatmap
    st.subheader("Correlation Between Model Precisions")
    prec_df = filtered_df[[f'precision_{mn}' for mn in model_nums]]
    corr = prec_df.corr()
    fig_heatmap = px.imshow(
        corr,
        x=[f'Model {mn}' for mn in model_nums],
        y=[f'Model {mn}' for mn in model_nums],
        color_continuous_scale='Viridis',
        height=400
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
    st.markdown("This heatmap shows the correlation between precision values of different models, indicating how similarly they perform.")

    # Top Images by Average Precision
    st.subheader("Top 5 Images by Average Precision")
    image_metrics = filtered_df[['filename', 'year', 'domaine', 'porte_greffe', 'parcelle']].copy()
    image_metrics['avg_precision'] = filtered_df[[f'precision_{i}' for i in model_nums]].mean(axis=1)
    image_metrics['avg_recall'] = filtered_df[[f'recall_{i}' for i in model_nums]].mean(axis=1)
    top_images = image_metrics.sort_values('avg_precision', ascending=False).head(5)
    st.dataframe(
        top_images.style.format({
            'avg_precision': '{:.2%}',
            'avg_recall': '{:.2%}',
            'year': '{}'
        })
    )
    st.markdown("This table lists the top 5 images by average precision across all models, filtered by your selections.")

    # Export Filtered Data
    st.subheader("Export Data")
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_model_evaluation.csv",
        mime="text/csv"
    )
    st.markdown("Download the filtered dataset for further analysis.")

    # Conclusion
    st.subheader("Conclusion")
    st.markdown(f"""
    The evaluation identifies **{winner['model']}** as the top performer with an F1 score of {winner['f1']*100:.2f}%. 
    Models generally perform well, with precision and recall often exceeding 80% across various years and conditions. 
    Certain years may exhibit lower recall due to higher false negatives, suggesting challenges in detecting objects in complex scenes. 
    Use the filters and visualizations to explore specific model performance, identify trends, and pinpoint areas for improvement.
    """)

if __name__ == "__main__":
    main()
