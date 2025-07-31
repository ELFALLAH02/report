import streamlit as st

def render_dashboard(metrics_df, model_nums, filtered_df):
    with st.expander("Dashboard Overview", expanded=True):
        winner = max(metrics_df.to_dict('records'), key=lambda x: x['f1'], default={'model': 'None', 'f1': 0})
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Models", len(model_nums))
        with col2:
            st.metric("Total Images", len(filtered_df))
        with col3:
            st.metric("Winning Model", winner['model'])
        with col4:
            st.metric("Top F1 Score", f"{winner['f1']*100:.2f}%")
        st.markdown("**Quick Insights**: This dashboard summarizes key metrics across all models. Expand sections below for detailed analysis.")