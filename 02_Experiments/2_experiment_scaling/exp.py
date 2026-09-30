# %%
import os
import time
import json
import csv
import numpy as np
import pandas as pd
import psutil
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

# %% [markdown]
# # 1. Custom Dataset: เครื่องมือชำแหละเวลา (Micro-benchmarking)

# %%
class ProfiledImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        # 1.1 Disk I/O (Access & Read)
        t_disk_start = time.time()
        path, target = self.samples[index]
        sample = self.loader(path)
        t_trans_start = time.time()
        
        # 1.2 CPU Transform (Decode, Resize, Augment)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        t_trans_end = time.time()
        
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        
        return sample, target, t_disk_start, t_trans_start, t_trans_end, worker_id

# %% [markdown]
# # 2. ฟังก์ชันจับเวลาระดับ Statement

# %%
def run_unbiased_epoch(dataloader, model, criterion, optimizer, device, is_train=True):
    if is_train:
        model.train()
    else:
        model.eval()
        
    cpu_metrics = {'Disk_Read': [], 'CPU_Transform': [], 'DataLoader_Wait': [], 'Disk_Read_mean': [], 'Disk_Read_max':[]}
    mem_metrics = {'RAM_GB': [], 'VRAM_GB': []}
    start_events = []
    end_events = []
    epoch_batch_logs = []
    
    # ตัวแปรสำหรับคำนวณ Accuracy และ Loss รวมของ Epoch
    total_loss = 0.0
    correct_preds = 0
    total_samples = 0
    
    torch.cuda.synchronize() # ซิงค์เพื่อตั้งจุดสตาร์ทของ Epoch
    epoch_start_time = time.perf_counter()
    epoch_start_ts_wall = time.time()
    epoch_start_ev = torch.cuda.Event(enable_timing=True)
    epoch_start_ev.record()
    end_time = epoch_start_time
    
    mode_name = "Train" if is_train else "Val  "
    pbar = tqdm(dataloader, desc=f"⏳ {mode_name}", leave=False, dynamic_ncols=True)
    process = psutil.Process(os.getpid())
    for images, targets, disk_starts, trans_starts, trans_ends, worker_ids in pbar:
        # --- เก็บฝั่ง CPU ---
        batch_start_ts = time.time()
        dataloader_wait = time.perf_counter() - end_time
        cpu_metrics['DataLoader_Wait'].append(dataloader_wait)
        
        disk_durations = trans_starts - disk_starts
        transform_durations = trans_ends - trans_starts
        
        cpu_metrics['Disk_Read'].append(disk_durations.sum().item())
        cpu_metrics['Disk_Read_mean'].append(disk_durations.mean().item())
        cpu_metrics['Disk_Read_max'].append(disk_durations.max().item())
        cpu_metrics['CPU_Transform'].append(transform_durations.sum().item())
        
        # --- เตรียมบัตรคิว GPU ---
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        
        start_ev.record()
        
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # --- การทำงานของโมเดล ---
        if is_train:
            outputs = model(images)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                outputs = model(images)
                loss = criterion(outputs, targets)
                
        end_ev.record()
        # เก็บข้อมูลเพื่อหาความแม่นยำ (ทำงานบน GPU/CPU ไม่บล็อก Pipeline หลัก)
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct_preds += predicted.eq(targets).sum().item()
        total_samples += images.size(0)
        
        start_events.append(start_ev)
        end_events.append(end_ev)
        
        disk_dur_ms = disk_durations.sum().item() * 1000.0
        trans_dur_ms = transform_durations.sum().item() * 1000.0
        
        # Relative timestamps from epoch wall clock
        disk_start_rel_ms = (disk_starts.min().item() - epoch_start_ts_wall) * 1000.0
        trans_end_rel_ms = (trans_ends.max().item() - epoch_start_ts_wall) * 1000.0
        worker_id = int(worker_ids[0].item())
        
        batch_info = {
            'batch_id': len(epoch_batch_logs),
            'disk_ms': disk_dur_ms,
            'disk_mean_ms': disk_durations.mean().item() * 1000.0,
            'disk_max_ms': disk_durations.max().item() * 1000.0,
            'trans_ms': trans_dur_ms,
            'dataloader_wait': dataloader_wait, 
            'batch_size': images.size(0), 
            'start_ts': batch_start_ts,
            'timestamp_relative': time.perf_counter() - epoch_start_time,
            
            # Precise profiling metrics
            'worker_id': worker_id,
            'disk_start_rel_ms': disk_start_rel_ms,
            'trans_end_rel_ms': trans_end_rel_ms,
        }
        epoch_batch_logs.append(batch_info)
        # --- เก็บ Resource ---
        mem_metrics['RAM_GB'].append(process.memory_info().rss / (1024 ** 3))
        if device == 'cuda':
            mem_metrics['VRAM_GB'].append(torch.cuda.memory_allocated() / (1024 ** 3))
        else:
            mem_metrics['VRAM_GB'].append(0.0)
        
        if is_train:
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
        end_time = time.perf_counter()

    # 🛑 จบ Epoch: สั่งบล็อกตรงนี้ครั้งเดียว เพื่อรอ GPU เคลียร์งานหยดสุดท้าย
    torch.cuda.synchronize()
    epoch_total_time = time.perf_counter() - epoch_start_time
    
    # --- คำนวณเวลาย้อนหลังของ GPU ทั้ง Epoch ---
    gpu_compute = []
    gpu_idle = []
    for i in range(len(start_events)):
        gpu_start_ms = epoch_start_ev.elapsed_time(start_events[i])
        gpu_end_ms = epoch_start_ev.elapsed_time(end_events[i])
        compute_ms = gpu_end_ms - gpu_start_ms
        
        epoch_batch_logs[i]['gpu_start_rel_ms'] = gpu_start_ms
        epoch_batch_logs[i]['gpu_end_rel_ms'] = gpu_end_ms
        epoch_batch_logs[i]['gpu_compute_ms'] = compute_ms
        
        if i == 0:
            idle_ms = gpu_start_ms
            gpu_util = None
        else:
            prev_gpu_end_ms = epoch_batch_logs[i-1]['gpu_end_rel_ms']
            idle_ms = max(0.0, gpu_start_ms - prev_gpu_end_ms)
            gpu_util = compute_ms / (compute_ms + idle_ms) if compute_ms + idle_ms > 0 else 0.0
            gpu_idle.append(idle_ms / 1000.0)
        epoch_batch_logs[i]['gpu_starvation_ms'] = idle_ms
        epoch_batch_logs[i]['gpu_utilize'] = gpu_util
        gpu_compute.append(compute_ms / 1000.0)
        
    epoch_loss = total_loss / total_samples
    epoch_acc = (correct_preds / total_samples) * 100.0

    gpu_busy_time = np.sum(gpu_compute[1:])
    gpu_starvation_time = np.sum(gpu_idle)
    denom = gpu_busy_time + gpu_starvation_time
    gpu_utilization_est = gpu_busy_time / denom if denom > 0 else 0.0
    gpu_utilization_est_all = np.sum(gpu_compute) / (np.sum(gpu_compute)+np.sum(gpu_idle))
    
    def safe_mean(metrics_list):
        if len(metrics_list) > 1:
            return np.mean(metrics_list[1:]) # ตัด Batch 0 ทิ้งตามปกติ
        elif len(metrics_list) == 1:
            return np.mean(metrics_list)     # ถ้ามีแค่ 1 Batch ก็ต้องยอมใช้ค่านั้น
        else:
            return 0.0
    # คืนค่าเป็นพจนานุกรม (Dictionary) ค่าเฉลี่ยต่อ Batch ภายใน Epoch นี้
    return {
        'Loss': epoch_loss,
        'Accuracy': epoch_acc,
        'Throughput_Img_Sec': total_samples / epoch_total_time,
        'Avg_Disk_Read': safe_mean(cpu_metrics['Disk_Read']),
        'Avg_Disk_Read_mean': safe_mean(cpu_metrics['Disk_Read_mean']),
        'Avg_Disk_Read_max': safe_mean(cpu_metrics['Disk_Read_max']),
        'Avg_Transform': safe_mean(cpu_metrics['CPU_Transform']),
        'Avg_Wait': safe_mean(cpu_metrics['DataLoader_Wait']),
        'Avg_GPU_Compute': safe_mean(gpu_compute),
        'Avg_GPU_Idle': safe_mean(gpu_idle),
        'Total_GPU_Compute': gpu_busy_time,
        'Total_GPU_Idle': gpu_starvation_time,
        'Avg_RAM_GB': safe_mean(mem_metrics['RAM_GB']),
        'Peak_RAM_GB': max(mem_metrics['RAM_GB']),
        'Avg_VRAM_GB': safe_mean(mem_metrics['VRAM_GB']),
        'Peak_VRAM':torch.cuda.max_memory_allocated() / 1024**3, 
        'GPU_Busy_Ratio_All': gpu_utilization_est_all, 
        'GPU_Busy_Ratio_NoWarmup': gpu_utilization_est, 
        'Epoch_Time_Sec': epoch_total_time, 
        'Total_GPU_Starvation_Sec': np.sum(gpu_idle), 
        'epoch_batch_logs': epoch_batch_logs
    }

# %% [markdown]
# # 3. Main Experiment & Plotting

# %%
def main_full_scale_training():
    # 🎯 ชี้ไปที่ Dataset จริง (โฟลเดอร์ต้องมีคลาสย่อยข้างใน)
    train_dir = '/home/mew/Desktop/mew/work/dataset/train'
    val_dir = '/home/mew/Desktop/mew/work/dataset/val'
    
    dry_run = False
    num_epochs = 5 # ปรับจำนวน Epoch ตามต้องการ
    batch_size = 128
    num_workers = 2 # 🎯 เปิด Worker เพื่อทดสอบ Large Scale vs OS Cache
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"กำลังเตรียมข้อมูล Full-scale (Workers: {num_workers})...")
    
    # Transform ปกติของการเทรน (เอา Resize โหดๆ ออกได้ถ้ารูปใหญ่พอที่จะเตะ OS Cache แล้ว)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = ProfiledImageFolder(train_dir, transform=train_transform)
    val_dataset = ProfiledImageFolder(val_dir, transform=val_transform)
    
    if dry_run:
        # 🎯 2. ท่าไม้ตาย Dry Run: หั่นข้อมูลมาแค่จำนวนนิดหน่อย (เช่น 128 รูป = 2 Batch)
        # ใช้ min() ป้องกันกรณีที่รูปในโฟลเดอร์มีน้อยกว่าที่เราจะเทส
        test_train_size = min(2048, len(train_dataset)) 
        test_val_size = min(1024, len(val_dataset))
        
        # สร้าง Subset เพื่อหลอก DataLoader
        train_dataset = Subset(train_dataset, range(test_train_size))
        val_dataset = Subset(val_dataset, range(test_val_size))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    model = models.resnet18(weights=None).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    
    train_num_batches = len(train_loader)
    val_num_batches = len(val_loader)
    # เตรียมไฟล์ CSV สำหรับเก็บประวัติการเทรนระดับ Epoch
    meta = f'{int(time.time())}_e{num_epochs}_bs{batch_size}_w{num_workers}_tb{train_num_batches}_vb{val_num_batches}' + str('_dry' if dry_run else '')
    if not os.path.exists(f'thesis_results_real/{meta}'):
        os.makedirs(f'thesis_results_real/{meta}')
    csv_filename = f'thesis_results_real/{meta}/epoch_result.csv'
    log_filename = f"thesis_results_real/{meta}/batch_logs.jsonl"
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Epoch', 
            'Phase', 
            'Loss', 
            'Accuracy', 
            'Throughput_Img_Sec', 
            'Disk_Read_Sec', 
            'Disk_Read_Mean_Sec', 
            'Disk_Read_Max_Sec', 
            'Transform_Sec', 
            'DataLoader_Wait_Sec', 
            'GPU_Compute_Sec', 
            'GPU_Idle_Starvation_Sec', 
            'Total_GPU_Compute_Sec',
            'Total_GPU_Idle_Sec',
            'RAM_GB', 
            'Peak_RAM_GB', 
            'VRAM_GB',
            'Peak_VRAM',
            'GPU_Busy_Ratio_All', 
            'GPU_Busy_Ratio_NoWarmup', 
            'Epoch_Time_Sec',
            'Total_GPU_Starvation_Sec'
        ])
        {
    }
        for epoch in range(num_epochs):
            print(f"\n[{time.strftime('%H:%M:%S')}] 🚀 เริ่มต้น Epoch {epoch+1}/{num_epochs}")
            
            # --- 1. Training Phase ---
            torch.cuda.reset_peak_memory_stats()
            print("กำลังรัน Training Phase (Full Dataset)...")
            train_metrics = run_unbiased_epoch(train_loader, model, criterion, optimizer, device, is_train=True)
            print(f"Train | Acc: {train_metrics['Accuracy']:.2f}% | Throughput: {train_metrics['Throughput_Img_Sec']:.1f} img/s | GPU Idle (Starvation): {train_metrics['Avg_GPU_Idle']:.4f} s/batch")
            
            writer.writerow([
                epoch+1, 
                'Train', 
                train_metrics['Loss'], 
                train_metrics['Accuracy'], 
                train_metrics['Throughput_Img_Sec'],
                train_metrics['Avg_Disk_Read'], 
                train_metrics['Avg_Disk_Read_mean'], 
                train_metrics['Avg_Disk_Read_max'], 
                train_metrics['Avg_Transform'], 
                train_metrics['Avg_Wait'],
                train_metrics['Avg_GPU_Compute'], 
                train_metrics['Avg_GPU_Idle'], 
                train_metrics['Total_GPU_Compute'], 
                train_metrics['Total_GPU_Idle'], 
                train_metrics['Avg_RAM_GB'], 
                train_metrics['Peak_RAM_GB'], 
                train_metrics['Avg_VRAM_GB'],
                train_metrics['Peak_VRAM'],
                train_metrics['GPU_Busy_Ratio_All'],
                train_metrics['GPU_Busy_Ratio_NoWarmup'],
                train_metrics['Epoch_Time_Sec'],
                train_metrics['Total_GPU_Starvation_Sec'],
            ])
            with open(log_filename, "a") as fl:
                for entry in train_metrics['epoch_batch_logs']:
                    entry['epoch'] = epoch+1
                    entry['phase'] = 'train'
                    fl.write(json.dumps(entry) + "\n")
            
            # --- 2. Validation Phase ---
            torch.cuda.reset_peak_memory_stats()
            print("กำลังรัน Validation Phase (Full Dataset)...")
            val_metrics = run_unbiased_epoch(val_loader, model, criterion, optimizer, device, is_train=False)
            print(f"Val   | Acc: {val_metrics['Accuracy']:.2f}% | Throughput: {val_metrics['Throughput_Img_Sec']:.1f} img/s | GPU Idle (Starvation): {val_metrics['Avg_GPU_Idle']:.4f} s/batch")
            
            writer.writerow([
                epoch+1, 
                'Validation', 
                val_metrics['Loss'], 
                val_metrics['Accuracy'], 
                val_metrics['Throughput_Img_Sec'],
                val_metrics['Avg_Disk_Read'], 
                val_metrics['Avg_Disk_Read_mean'], 
                val_metrics['Avg_Disk_Read_max'], 
                val_metrics['Avg_Transform'], 
                val_metrics['Avg_Wait'],
                val_metrics['Avg_GPU_Compute'], 
                val_metrics['Avg_GPU_Idle'], 
                val_metrics['Total_GPU_Compute'], 
                val_metrics['Total_GPU_Idle'], 
                val_metrics['Avg_RAM_GB'], 
                val_metrics['Peak_RAM_GB'], 
                val_metrics['Avg_VRAM_GB'],
                val_metrics['Peak_VRAM'],
                val_metrics['GPU_Busy_Ratio_All'],
                val_metrics['GPU_Busy_Ratio_NoWarmup'],
                val_metrics['Epoch_Time_Sec'],
                val_metrics['Total_GPU_Starvation_Sec'],
            ])
            with open(log_filename, "a") as fl:
                for entry in val_metrics['epoch_batch_logs']:
                    entry['epoch'] = epoch+1
                    entry['phase'] = 'val'
                    fl.write(json.dumps(entry) + "\n")
            
            f.flush() # บังคับเขียนลงไฟล์ทันทีเผื่อสคริปต์ดับกลางทาง

    print(f"\n✅ การเทรนเสร็จสมบูรณ์! ผลลัพธ์ถูกบันทึกไว้ใน {csv_filename}")

# %%
main_full_scale_training()


