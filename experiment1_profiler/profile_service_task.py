#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, math, os, platform, shlex, socket, statistics, subprocess, sys, threading, time, traceback, uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import psutil

class GPUReader:
    def __init__(self, index=0):
        self.index=index; self.backend=None; self.handle=None; self.nvml=None; self.gpu_name=None
        try:
            import pynvml
            pynvml.nvmlInit(); self.nvml=pynvml
            self.handle=pynvml.nvmlDeviceGetHandleByIndex(index)
            n=pynvml.nvmlDeviceGetName(self.handle)
            self.gpu_name=n.decode() if isinstance(n,bytes) else str(n)
            self.backend='pynvml'; return
        except Exception: pass
        try:
            q=subprocess.run(['nvidia-smi',f'--id={index}','--query-gpu=name','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=5)
            self.gpu_name=q.stdout.strip().splitlines()[0]; self.backend='nvidia-smi'
        except Exception: pass
    def read(self):
        out=dict(gpu_util_pct=None,gpu_mem_used_mb=None,gpu_mem_total_mb=None,gpu_temp_c=None,gpu_power_w=None)
        if self.backend=='pynvml':
            try:
                u=self.nvml.nvmlDeviceGetUtilizationRates(self.handle); m=self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
                t=self.nvml.nvmlDeviceGetTemperature(self.handle,self.nvml.NVML_TEMPERATURE_GPU)
                try: p=self.nvml.nvmlDeviceGetPowerUsage(self.handle)/1000.0
                except Exception: p=None
                out.update(gpu_util_pct=float(u.gpu),gpu_mem_used_mb=m.used/1024**2,gpu_mem_total_mb=m.total/1024**2,gpu_temp_c=float(t),gpu_power_w=p)
            except Exception: pass
        elif self.backend=='nvidia-smi':
            try:
                q=subprocess.run(['nvidia-smi',f'--id={self.index}','--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=5)
                v=[x.strip() for x in q.stdout.strip().splitlines()[0].split(',')]
                out.update(gpu_util_pct=float(v[0]),gpu_mem_used_mb=float(v[1]),gpu_mem_total_mb=float(v[2]),gpu_temp_c=float(v[3]),gpu_power_w=float(v[4]))
            except Exception: pass
        return out
    def close(self):
        if self.backend=='pynvml':
            try:self.nvml.nvmlShutdown()
            except Exception:pass

def tree(proc):
    try:return [proc]+proc.children(recursive=True)
    except Exception:return [proc]

def rss_mb(proc):
    total=0
    for p in tree(proc):
        try: total += p.memory_info().rss
        except Exception: pass
    return total/1024**2

def proc_cpu(proc):
    total=0.0
    for p in tree(proc):
        try: total += p.cpu_percent(interval=None)
        except Exception: pass
    return total

@dataclass
class Sample:
    timestamp_utc:str; elapsed_sec:float; system_cpu_pct:float; system_ram_used_mb:float; system_ram_available_mb:float
    process_cpu_pct:float; process_rss_mb:float; gpu_util_pct:Optional[float]; gpu_mem_used_mb:Optional[float]
    gpu_mem_total_mb:Optional[float]; gpu_temp_c:Optional[float]; gpu_power_w:Optional[float]

class Sampler:
    def __init__(self,proc,gpu_index,interval):
        self.proc=proc; self.interval=interval; self.gpu=GPUReader(gpu_index); self.samples=[]; self.stop_event=threading.Event(); self.t0=time.perf_counter(); self.thread=threading.Thread(target=self._loop,daemon=True)
        psutil.cpu_percent(None)
        for p in tree(proc):
            try:p.cpu_percent(None)
            except Exception:pass
    def start(self):self.thread.start()
    def stop(self):self.stop_event.set(); self.thread.join(timeout=max(2,self.interval*4)); self.gpu.close()
    def _loop(self):
        while not self.stop_event.is_set():
            try:
                vm=psutil.virtual_memory(); g=self.gpu.read()
                self.samples.append(Sample(datetime.now(timezone.utc).isoformat(),time.perf_counter()-self.t0,float(psutil.cpu_percent(None)),vm.used/1024**2,vm.available/1024**2,proc_cpu(self.proc),rss_mb(self.proc),g['gpu_util_pct'],g['gpu_mem_used_mb'],g['gpu_mem_total_mb'],g['gpu_temp_c'],g['gpu_power_w']))
            except Exception: pass
            self.stop_event.wait(self.interval)

def vals(samples,name):
    a=[]
    for s in samples:
        v=getattr(s,name)
        if isinstance(v,(int,float)) and math.isfinite(v):a.append(float(v))
    return a

def mean(a): return statistics.mean(a) if a else None
def peak(a): return max(a) if a else None
def p95(a):
    if not a:return None
    b=sorted(a); return b[max(0,min(len(b)-1,math.ceil(.95*len(b))-1))]

def artifact_size(path):
    if not path:return None
    p=Path(path)
    if not p.exists():return None
    if p.is_file():return p.stat().st_size/1024**2
    return sum(x.stat().st_size for x in p.rglob('*') if x.is_file())/1024**2

def metric_file(path):
    if not path:return {}
    p=Path(path)
    if not p.exists():return {'metric_file_missing':True}
    try:
        x=json.loads(p.read_text())
        return x if isinstance(x,dict) else {'value':x}
    except Exception as e:return {'metric_file_error':str(e)}

def append_csv(path,row):
    exists=path.exists(); fields=list(row)
    with path.open('a',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields)
        if not exists:w.writeheader()
        w.writerow(row)

def parse():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True); p.add_argument('--task',required=True,choices=['update','evaluate','package','deploy','validate','reactivate','inference'])
    p.add_argument('--host',default=socket.gethostname()); p.add_argument('--target-edge',default=''); p.add_argument('--gpu-index',type=int,default=0)
    p.add_argument('--sample-interval',type=float,default=.5); p.add_argument('--artifact'); p.add_argument('--metric-file')
    p.add_argument('--output-dir',default='/workspace/sc26_rebuttal/outputs/experiment1'); p.add_argument('--run-id'); p.add_argument('--notes',default='')
    p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args();
    if a.command and a.command[0]=='--':a.command=a.command[1:]
    if not a.command:p.error('Put the workload command after --')
    return a

def main():
    a=parse(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    rid=a.run_id or f'{a.model}_{a.task}_{uuid.uuid4().hex[:10]}'; rdir=out/'runs'/rid; rdir.mkdir(parents=True,exist_ok=True)
    stdout=rdir/'stdout.log'; stderr=rdir/'stderr.log'; samples_csv=rdir/'samples.csv'; result_json=rdir/'result.json'
    g0=GPUReader(a.gpu_index); gpu_name=g0.gpu_name; gpu_backend=g0.backend; g0.close()
    d0=psutil.disk_io_counters(); n0=psutil.net_io_counters(); t0=datetime.now(timezone.utc); q0=time.perf_counter(); sampler=None; rc=None; fail=''
    try:
        with stdout.open('w') as so, stderr.open('w') as se:
            proc=subprocess.Popen(a.command,stdout=so,stderr=se,env=os.environ.copy(),cwd=os.getcwd())
            sampler=Sampler(psutil.Process(proc.pid),a.gpu_index,a.sample_interval); sampler.start(); rc=proc.wait()
    except Exception as e:
        fail=f'{type(e).__name__}: {e}'; stderr.write_text(traceback.format_exc())
    finally:
        if sampler:sampler.stop()
    runtime=time.perf_counter()-q0; t1=datetime.now(timezone.utc); d1=psutil.disk_io_counters(); n1=psutil.net_io_counters(); ss=sampler.samples if sampler else []
    if ss:
        with samples_csv.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(asdict(ss[0]))); w.writeheader(); [w.writerow(asdict(s)) for s in ss]
    result={
      'run_id':rid,'model':a.model,'task':a.task,'host':a.host,'target_edge':a.target_edge,'gpu_index':a.gpu_index,'gpu_name':gpu_name,'gpu_reader_backend':gpu_backend,
      'command':shlex.join(a.command),'start_time_utc':t0.isoformat(),'end_time_utc':t1.isoformat(),'runtime_sec':runtime,'sample_interval_sec':a.sample_interval,'n_samples':len(ss),
      'system_cpu_mean_pct':mean(vals(ss,'system_cpu_pct')),'system_cpu_peak_pct':peak(vals(ss,'system_cpu_pct')),'system_cpu_p95_pct':p95(vals(ss,'system_cpu_pct')),
      'process_cpu_mean_pct':mean(vals(ss,'process_cpu_pct')),'process_cpu_peak_pct':peak(vals(ss,'process_cpu_pct')),'process_rss_peak_mb':peak(vals(ss,'process_rss_mb')),
      'system_ram_used_peak_mb':peak(vals(ss,'system_ram_used_mb')),'system_ram_available_min_mb':min(vals(ss,'system_ram_available_mb')) if vals(ss,'system_ram_available_mb') else None,
      'gpu_util_mean_pct':mean(vals(ss,'gpu_util_pct')),'gpu_util_peak_pct':peak(vals(ss,'gpu_util_pct')),'gpu_util_p95_pct':p95(vals(ss,'gpu_util_pct')),
      'gpu_mem_used_peak_mb':peak(vals(ss,'gpu_mem_used_mb')),'gpu_mem_total_mb':vals(ss,'gpu_mem_total_mb')[0] if vals(ss,'gpu_mem_total_mb') else None,
      'gpu_temp_peak_c':peak(vals(ss,'gpu_temp_c')),'gpu_power_mean_w':mean(vals(ss,'gpu_power_w')),'gpu_power_peak_w':peak(vals(ss,'gpu_power_w')),
      'disk_read_mb':(d1.read_bytes-d0.read_bytes)/1024**2 if d0 and d1 else None,'disk_write_mb':(d1.write_bytes-d0.write_bytes)/1024**2 if d0 and d1 else None,
      'network_rx_mb':(n1.bytes_recv-n0.bytes_recv)/1024**2 if n0 and n1 else None,'network_tx_mb':(n1.bytes_sent-n0.bytes_sent)/1024**2 if n0 and n1 else None,
      'artifact_path':a.artifact or '','artifact_size_mb':artifact_size(a.artifact),'return_code':rc,'success':bool(rc==0 and not fail),'failure_reason':fail,'notes':a.notes,
      'python_version':platform.python_version(),'platform':platform.platform()
    }
    for k,v in metric_file(a.metric_file).items(): result[f'metric_{k}']=v if isinstance(v,(str,int,float,bool)) or v is None else json.dumps(v)
    result_json.write_text(json.dumps(result,indent=2));
    with (out/'experiment1_task_profiles.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
    append_csv(out/'experiment1_task_profiles.csv',result)
    print(f'Run ID: {rid}\nModel: {a.model}\nTask: {a.task}\nGPU: {gpu_name}\nRuntime: {runtime:.3f}s\nSuccess: {result["success"]}\nSamples: {len(ss)}\nResult: {result_json}')
    sys.exit(0 if result['success'] else 1)

if __name__=='__main__':main()
