import streamlit as st
import plotly.express as px
import pandas as pd
@st.cache_data


def render_performance_by_year(filtered_df, model_nums):
    with st.expander("Performance by Year", expanded=False):
        # Ensure year is string for consistent display
        filtered_df['year'] = filtered_df['year'].astype(str)
        # Debug: Show available years
        df = filtered_df.copy()
        df['year'] = df['year'].astype(str)

        
        year_data = filtered_df.groupby('year').agg({
        year_data = df.groupby('year').agg({
            f'precision_{i}': 'mean' for i in model_nums
        }).reset_index()
        year_data['avg_precision'] = year_data[[f'precision_{i}' for i in model_nums]].mean(axis=1)
        year_data['avg_recall'] = filtered_df.groupby('year').agg({
        year_data['avg_recall'] = df.groupby('year').agg({
            f'recall_{i}': 'mean' for i in model_nums
        }).reset_index()[[f'recall_{i}' for i in model_nums]].mean(axis=1)
        

        fig_scatter = px.scatter(
            year_data,
            x='avg_precision',
            y='avg_recall',
            color=year_data['year'].astype(str),  # Treat year as categorical
            text='year',
            labels={'avg_precision': 'Average Precision', 'avg_recall': 'Average Recall'},
            color_discrete_sequence=px.colors.qualitative.T10
        )
        fig_scatter.update_traces(
            textposition='top center',
            marker_size=8,
            textfont=dict(size=10)
        )
        fig_scatter.update_layout(
            xaxis_tickformat=".0%",
            yaxis_tickformat=".0%",
            height=500,
            margin=dict(t=50, b=50),
            showlegend=True
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown("This scatter plot shows average precision vs. recall for each year, filtered by your selections.", help="Each point represents a year’s average performance across all models.")
        st.markdown(
            "This scatter plot shows average precision vs. recall for each year, filtered by your selections.",
            help="Each point represents a year’s average performance across all models."
        )
