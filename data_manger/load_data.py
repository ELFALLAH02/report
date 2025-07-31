import streamlit as st
import pandas as pd
import os
import glob
import re
@st.cache_data
def load_data(data_folder):
    try:
        csv_files = glob.glob(os.path.join(data_folder, "eval_model_*_Sheet1.csv"))
        if not csv_files:
            st.error(f"No CSV files found in {data_folder}")
            return None, []

        merged_df = None
        model_nums = []
        common_cols = ['filename', 'year', 'domaine', 'porte_greffe', 'parcelle']

        for file in csv_files:
            match = re.search(r'eval_model_(\d+)_Sheet1\.csv', os.path.basename(file))
            if not match:
                st.warning(f"Skipping file with invalid name format: {file}")
                continue
            model_num = int(match.group(1))
            if model_num == 18:
                continue
            model_nums.append(model_num)
            
            df = pd.read_csv(file)
            df.columns = df.columns.str.strip().str.lower().str.replace('"', '').str.replace('porte-greffe', 'porte_greffe')
            
            for col in ['true_count', 'detect_count', 'tp', 'fp', 'fn']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            for col in ['precision', 'recall']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            df = df.rename(columns={
                'precision': f'precision_{model_num}',
                'recall': f'recall_{model_num}',
                'tp': f'tp_{model_num}',
                'fp': f'fp_{model_num}',
                'fn': f'fn_{model_num}',
                'true_count': f'true_count_{model_num}',
                'detect_count': f'detect_count_{model_num}'
            })
            
            if 'compagnie' in df.columns:
                df = df.rename(columns={'compagnie': 'year'})
            
            if merged_df is None:
                merged_df = df
            else:
                merge_cols = [col for col in common_cols if col in df.columns and col in merged_df.columns]
                if not merge_cols:
                    st.error(f"No common columns to merge on for file: {file}")
                    return None, []
                try:
                    merged_df = merged_df.merge(df, on=merge_cols, how='outer', suffixes=(None, f'_model_{model_num}'))
                except Exception as e:
                    st.error(f"Error merging file {file}: {str(e)}")
                    return None, []
        
        if merged_df is None:
            st.error("No valid data found in CSV files")
            return None, []
        
        columns = merged_df.columns
        for col in columns:
            if col.endswith('_x') or col.endswith('_y') or any(col.endswith(f'_model_{mn}') for mn in model_nums):
                merged_df = merged_df.drop(columns=col)
        
        return merged_df, sorted(model_nums)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, []