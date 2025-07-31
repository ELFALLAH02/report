import streamlit as st
import plotly.express as px
import pandas as pd

def render_precision_trends(filtered_df, model_nums):
    with st.expander("Precision Trends Over Years", expanded=False):
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
            height=400,
            color_discrete_sequence=px.colors.qualitative.T10
        )
        fig_line_precision.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig_line_precision, use_container_width=True)
        st.markdown("This line chart shows how average precision for each model changes across years.")