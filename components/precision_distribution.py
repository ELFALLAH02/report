import streamlit as st
import plotly.express as px
import pandas as pd
@st.cache_data
def render_precision_distribution(filtered_df, model_nums):
    with st.expander("Precision Distribution Across Models", expanded=False):
        precision_cols = [f'precision_{mn}' for mn in model_nums]
        melt_df_precision = filtered_df.melt(id_vars=['filename'], value_vars=precision_cols, var_name='model', value_name='precision')
        melt_df_precision['model'] = melt_df_precision['model'].str.replace('precision_', 'Model ')
        fig_box_precision = px.box(
            melt_df_precision,
            x='model',
            y='precision',
            labels={'precision': 'Precision'},
            height=400,
            color_discrete_sequence=['#1B9E77']
        )
        fig_box_precision.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig_box_precision, use_container_width=True)
        st.markdown("This box plot shows the distribution of precision for each model across all images.", help="Box plots show median, quartiles, and outliers for precision.")