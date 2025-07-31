import streamlit as st
import pandas as pd

def render_performance_by_context(filtered_df, model_nums):
    with st.expander("Performance by Domaine and Porte Greffe", expanded=False):
        st.markdown("Analyze model performance across different domaines and porte greffes.")
        context_cols = ['domaine', 'porte_greffe']
        for context in context_cols:
            context_data = filtered_df.groupby(context).agg({
                f'precision_{i}': 'mean' for i in model_nums
            }).reset_index()
            context_data['avg_precision'] = context_data[[f'precision_{i}' for i in model_nums]].mean(axis=1)
            context_data['avg_recall'] = filtered_df.groupby(context).agg({
                f'recall_{i}': 'mean' for i in model_nums
            }).reset_index()[[f'recall_{i}' for i in model_nums]].mean(axis=1)
            context_data = context_data[[context, 'avg_precision', 'avg_recall']].sort_values('avg_precision', ascending=False)
            st.subheader(f"Performance by {context.capitalize()}")
            st.dataframe(context_data.style.format({
                'avg_precision': '{:.2%}',
                'avg_recall': '{:.2%}'
            }))
            st.markdown(f"This table shows average precision and recall for each {context}, helping identify conditions where models perform best or worst.")