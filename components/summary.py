import streamlit as st

def render_summary(metrics_df, years):
    with st.expander("Summary", expanded=True):
        records = metrics_df.to_dict('records')
        st.markdown(f"""
        This report evaluates {len(records)} machine learning models for object detection in citrus groves across {len(years)} years ({', '.join(years)}). 
        Metrics include:
        - **True Positives (TP)**: Correctly detected objects.
        - **False Positives (FP)**: Incorrectly detected objects.
        - **False Negatives (FN)**: Missed objects.
        - **Precision**: Proportion of detections that were correct (TP / (TP + FP)).
        - **Recall**: Proportion of actual objects detected (TP / (TP + FN)).
        - **F1 Score**: Harmonic mean of precision and recall.
        """)
        winner = max(records, key=lambda x: x['f1'], default={'model': 'None', 'f1': 0})
        st.markdown(f"The **winning model** is {winner['model']} with an F1 score of {winner['f1']*100:.2f}%.")
        st.markdown(f"""
        **Key Observations**:
        - **Highest Precision**: {max(records, key=lambda x: x['avg_precision'], default={'model': 'None', 'avg_precision': 0})['model']} ({max(records, key=lambda x: x['avg_precision'], default={'model': 'None', 'avg_precision': 0})['avg_precision']*100:.2f}%)
        - **Highest Recall**: {max(records, key=lambda x: x['avg_recall'], default={'model': 'None', 'avg_recall': 0})['model']} ({max(records, key=lambda x: x['avg_recall'], default={'model': 'None', 'avg_recall': 0})['avg_recall']*100:.2f}%)
        - Some years may show lower recall due to higher false negatives, indicating challenges in complex scenes.
        """)