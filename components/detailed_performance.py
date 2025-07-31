import streamlit as st
import pandas as pd

def render_detailed_performance(selected_model, model_metrics, metrics_df):
    with st.expander("Detailed Model Performance", expanded=False):
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