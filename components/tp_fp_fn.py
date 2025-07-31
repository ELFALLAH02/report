import streamlit as st
import plotly.express as px
import pandas as pd

def render_tp_fp_fn(filtered_df, model_nums):
    with st.expander("True Positives, False Positives, and False Negatives", expanded=False):
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
                'True Positives': '#1B9E77',
                'False Positives': '#D95F02',
                'False Negatives': '#7570B3'
            }
        )
        st.plotly_chart(fig_stacked, use_container_width=True)
        st.markdown("This stacked bar chart shows the total counts of true positives, false positives, and false negatives for each model.", help="High FPs or FNs may indicate model weaknesses.")