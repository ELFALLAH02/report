import streamlit as st

def render_help_section():
    with st.expander("Help & Documentation", expanded=False):
        st.markdown("""
        **How to Use This Report**:
        - Use the **Filters** to narrow down data by model, year, domaine, porte greffe, or parcelle.
        - Explore visualizations to understand model performance trends and distributions.
        - Check the **Error Analysis** section to identify images with high false positives/negatives.
        - Download charts as PNGs or the filtered dataset as CSV for further analysis.
        
        **Metric Definitions**:
        - **Precision**: Proportion of detections that were correct.
        - **Recall**: Proportion of actual objects detected.
        - **F1 Score**: Balances precision and recall for overall performance.
        - **True Positives (TP)**: Correct detections.
        - **False Positives (FP)**: Incorrect detections.
        - **False Negatives (FN)**: Missed detections.
        
        For support, contact your data science team.
        """)