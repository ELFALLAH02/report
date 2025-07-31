import streamlit as st
import plotly.express as px
import pandas as pd

def render_correlation_heatmap(filtered_df, model_nums):
    with st.expander("Correlation Between Model Precisions", expanded=False):
        st.subheader("Model Precision Correlation")
        st.markdown(
            "This heatmap shows how similarly models perform based on their precision values. "
            "High correlations (close to 1) indicate models behave similarly, while low or negative correlations suggest differing performance."
        )

        # Check if there are enough models for correlation
        if len(model_nums) < 2:
            st.warning("At least two models are required to compute correlations.")
            return

        # Prepare data
        prec_df = filtered_df[[f'precision_{mn}' for mn in model_nums]]
        if prec_df.empty or prec_df.shape[1] < 2:
            st.warning("Insufficient data to compute correlations.")
            return

        # Compute correlation
        corr = prec_df.corr()

        # Toggle for annotations
        show_annotations = st.checkbox("Show Correlation Values", value=False, key="corr_annotations")

        # Create heatmap
        fig_heatmap = px.imshow(
            corr,
            x=[f'Model {mn}' for mn in model_nums],
            y=[f'Model {mn}' for mn in model_nums],
            color_continuous_scale='Viridis',
            height=500,
            title="Precision Correlation Heatmap",
            zmin=-1,
            zmax=1
        )

        # Add annotations if enabled
        if show_annotations:
            annotations = [
                dict(
                    x=i,
                    y=j,
                    text=f"{corr.iloc[j, i]:.2f}",
                    showarrow=False,
                    font=dict(
                        color="white" if abs(corr.iloc[j, i]) > 0.5 else "black",
                        size=12
                    )
                )
                for i in range(len(corr))
                for j in range(len(corr))
            ]
            fig_heatmap.update_layout(annotations=annotations)

        # Update layout for clarity
        fig_heatmap.update_layout(
            xaxis_title="Model",
            yaxis_title="Model",
            xaxis_tickangle=45,
            margin=dict(t=100, b=100),
            coloraxis_colorbar_title="Correlation"
        )

        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Interpretation
        st.markdown(
            """
            **Interpretation**:
            - **High Correlations (> 0.8)**: Models perform similarly, possibly due to similar architectures or training data.
            - **Low/Negative Correlations (< 0.2)**: Models differ significantly, which may indicate diverse strengths or weaknesses.
            - Use this to identify redundant models or unique performers.
            """
        )