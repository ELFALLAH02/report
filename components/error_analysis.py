import streamlit as st
import pandas as pd

def render_error_analysis(filtered_df, model_nums):
    with st.expander("Error Analysis", expanded=False):
        st.markdown("Identify images with high false positives or false negatives for further investigation.")
        error_data = filtered_df[['filename', 'year', 'domaine', 'porte_greffe', 'parcelle']].copy()
        for mn in model_nums:
            error_data[f'fp_{mn}'] = filtered_df[f'fp_{mn}']
            error_data[f'fn_{mn}'] = filtered_df[f'fn_{mn}']
        error_data['avg_fp'] = error_data[[f'fp_{mn}' for mn in model_nums]].mean(axis=1)
        error_data['avg_fn'] = error_data[[f'fn_{mn}' for mn in model_nums]].mean(axis=1)
        high_errors = error_data[error_data[['avg_fp', 'avg_fn']].max(axis=1) > error_data[['avg_fp', 'avg_fn']].quantile(0.95).max()]
        st.dataframe(high_errors[['filename', 'year', 'domaine', 'porte_greffe', 'parcelle', 'avg_fp', 'avg_fn']].style.format({
            'avg_fp': '{:.1f}',
            'avg_fn': '{:.1f}'
        }))
        st.markdown("This table lists images with unusually high false positives or false negatives (top 5% of errors), indicating potential challenges in detection.")