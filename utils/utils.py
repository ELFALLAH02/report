import streamlit as st
import pandas as pd
import os
import glob
import re
import numpy as np
from io import BytesIO
import base64

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

def calculate_model_metrics(df, model_nums):
    model_metrics = []
    for mn in model_nums:
        prec_col = f'precision_{mn}'
        rec_col = f'recall_{mn}'
        tp_col = f'tp_{mn}'
        fp_col = f'fp_{mn}'
        fn_col = f'fn_{mn}'
        
        valid_data = df[df[prec_col] > 0]
        avg_precision = valid_data[prec_col].mean() if not valid_data.empty else 0
        avg_recall = valid_data[rec_col].mean() if not valid_data.empty else 0
        total_tp = df[tp_col].sum()
        total_fp = df[fp_col].sum()
        total_fn = df[fn_col].sum()
        f1 = (2 * avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        
        model_metrics.append({
            'model': f'Model {mn}',
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'f1': f1,
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn,
            'data': df[['filename', 'year', 'domaine', 'porte_greffe', 'parcelle', prec_col, rec_col, tp_col, fp_col, fn_col]].copy()
        })
    return model_metrics

