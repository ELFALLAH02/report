import streamlit as st
import plotly.express as px
import pandas as pd

def render_recall_distribution(filtered_df, model_nums):
    with st.expander("Recall Distribution Across Models", expanded=False):
        recall_cols = [f'recall_{mn}' for mn in model_nums]
        melt_df_recall = filtered_df.melt(id_vars=['filename'], value_vars=recall_cols, var_name='model', value_name='recall')
        melt_df_recall['model'] = melt_df_recall['model'].str.replace('recall_', 'Model ')
        fig_box_recall = px.box(
            melt_df_recall,
            x='model',
            y='recall',
            labels={'recall': 'Recall'},
            height=400,
            color_discrete_sequence=['#D95F02']
        )
        fig_box_recall.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig_box_recall, use_container_width=True)
        st.markdown("This box plot shows the distribution of recall for each model across all images.", help="Box plots show median, quartiles, and outliers for recall.")