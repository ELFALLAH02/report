import streamlit as st
import plotly.graph_objects as go

def render_model_performance(metrics_df):
    with st.expander("Model Performance Overview", expanded=True):
        fig_bar = go.Figure(data=[
            go.Bar(name='Average Precision', x=metrics_df['model'], y=metrics_df['avg_precision'], marker_color='#1B9E77'),
            go.Bar(name='Average Recall', x=metrics_df['model'], y=metrics_df['avg_recall'], marker_color='#D95F02')
        ])
        fig_bar.update_layout(
            barmode='group',
            xaxis_title="Model",
            yaxis_title="Score",
            yaxis_tickformat=".0%",
            xaxis_tickangle=-45,
            height=400,
            margin=dict(b=150),
            colorway=['#1B9E77', '#D95F02']
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown("This bar chart compares average precision and recall across models, filtered by your selections.", help="Precision measures detection accuracy; recall measures detection completeness.")