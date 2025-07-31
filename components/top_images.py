import streamlit as st
import pandas as pd

def render_top_images(filtered_df, model_nums):
    with st.expander("Top 5 Images by Average Precision", expanded=False):
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