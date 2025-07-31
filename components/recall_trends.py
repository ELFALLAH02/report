import streamlit as st
import plotly.express as px
import pandas as pd

def render_recall_trends(filtered_df, model_nums):
    with st.expander("Recall Trends Over Years", expanded=False):
        @st.cache_data
        def calculate_recall_trends(df, model_nums):
            long_df_recall = []
            for mn in model_nums:
                rec_col = f'recall_{mn}'
                if rec_col in df.columns:
                    temp_df = df[df[rec_col] > 0][['year', rec_col]].groupby('year')[rec_col].mean().reset_index()
                    if not temp_df.empty:
                        temp_df = temp_df.rename(columns={rec_col: 'avg_recall'})
                        temp_df['model'] = f'Model {mn}'
                        long_df_recall.append(temp_df)
            if long_df_recall:
                recall_trend_df = pd.concat(long_df_recall)
                recall_trend_df['year'] = recall_trend_df['year'].astype(str)
                return recall_trend_df
            return pd.DataFrame(columns=['year', 'avg_recall', 'model'])
        recall_trend_df = calculate_recall_trends(filtered_df, model_nums)
        if not recall_trend_df.empty:
            fig_line_recall = px.line(
                recall_trend_df,
                x='year',
                y='avg_recall',
                color='model',
                labels={'avg_recall': 'Average Recall', 'year': 'Year'},
                height=400,
                markers=True,
                color_discrete_sequence=px.colors.qualitative.T10
            )
            fig_line_recall.update_layout(
                yaxis_tickformat=".0%",
                showlegend=True,
                legend=dict(yanchor="top", y=1.1, xanchor="left", x=0)
            )
            st.plotly_chart(fig_line_recall, use_container_width=True)
            st.markdown("This line chart shows how average recall for each model changes across years (excluding recall <= 0).")
        else:
            st.warning("No data available for the recall trends chart based on current filters.")