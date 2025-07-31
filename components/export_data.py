import streamlit as st
import pandas as pd

def render_export_data(filtered_df):
    with st.expander("Export Data", expanded=False):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name="filtered_model_evaluation.csv",
            mime="text/csv"
        )
        st.markdown("Download the filtered dataset for further analysis.")