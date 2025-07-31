import streamlit as st
import pandas as pd

def render_model_ranking(metrics_df):
    with st.expander("Model Ranking", expanded=True):
        st.markdown("Sort the table below to compare model performance across key metrics.")
        ranking_df = metrics_df[['model', 'f1', 'avg_precision', 'avg_recall', 'total_tp', 'total_fp', 'total_fn']].copy()
        ranking_df['f1'] = ranking_df['f1'].map('{:.2%}'.format)
        ranking_df['avg_precision'] = ranking_df['avg_precision'].map('{:.2%}'.format)
        ranking_df['avg_recall'] = ranking_df['avg_recall'].map('{:.2%}'.format)
        st.dataframe(ranking_df, use_container_width=True)