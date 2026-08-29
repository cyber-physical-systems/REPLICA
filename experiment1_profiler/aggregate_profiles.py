#!/usr/bin/env python3
from pathlib import Path
import argparse, json, pandas as pd
p=argparse.ArgumentParser(); p.add_argument('--input',default='/workspace/sc26_rebuttal/outputs/experiment1/experiment1_task_profiles.jsonl'); p.add_argument('--output-dir',default='/workspace/sc26_rebuttal/outputs/experiment1'); a=p.parse_args()
out=Path(a.output_dir); rows=[json.loads(x) for x in Path(a.input).read_text().splitlines() if x.strip()]; df=pd.DataFrame(rows); df.to_csv(out/'experiment1_all_runs.csv',index=False)
metrics=[c for c in ['runtime_sec','system_cpu_mean_pct','system_cpu_peak_pct','process_rss_peak_mb','gpu_util_mean_pct','gpu_util_peak_pct','gpu_mem_used_peak_mb','gpu_power_mean_w','disk_read_mb','disk_write_mb','network_rx_mb','network_tx_mb','artifact_size_mb'] if c in df.columns]
ok=df[df.success==True].copy(); group=['model','task','host','gpu_name']
if len(ok):
    s=ok.groupby(group,dropna=False)[metrics].agg(['count','mean','median','std','min','max']); s.columns=['_'.join(c) for c in s.columns]; s.reset_index().to_csv(out/'experiment1_summary.csv',index=False)
comp=df.groupby(group,dropna=False).agg(runs=('run_id','count'),successful_runs=('success','sum'),success_rate=('success','mean')).reset_index(); comp['compatible']=comp.success_rate>0; comp.to_csv(out/'experiment1_compatibility_matrix.csv',index=False)
print('Saved summaries to',out)
