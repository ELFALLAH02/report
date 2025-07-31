import streamlit as st

def render_conclusion(metrics_df):
    with st.expander("Conclusion", expanded=True):
        winner = max(metrics_df.to_dict('records'), key=lambda x: x['f1'], default={'model': 'None', 'f1': 0})
        st.markdown(f"""
        The evaluation identifies **{winner['model']}** as the top performer with an F1 score of {winner['f1']*100:.2f}%. 
        Models generally perform well, with precision and recall often exceeding 80% across various years and conditions. 
        Certain years or domaines may exhibit lower recall due to higher false negatives, suggesting challenges in complex scenes (see Error Analysis). 
        Use the filters and visualizations to explore specific model performance, identify trends, and pinpoint areas for improvement.
        """)