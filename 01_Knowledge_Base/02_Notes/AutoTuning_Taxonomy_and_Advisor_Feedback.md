---
tags:
  - taxonomy
  - research-idea
  - advisor-feedback
  - auto-tuning
aliases:
  - AutoTuning_Taxonomy
---

# 📐 Taxonomy of ML Data Loading Auto-Tuning & Advisor Feedback Analysis

_(บันทึกจากการหารือกับอาจารย์ที่ปรึกษาล่าสุด - กันยายน 2026)_

เอกสารนี้จัดทำขึ้นเพื่อตอบโจทย์ 3 ประเด็นหลักที่อาจารย์ได้ให้คำแนะนำ:

1. การทำ **Taxonomy** แบ่งประเภทการ Auto-tune ใน ML Data Pipeline และการระบุตำแหน่งงานวิจัยของเรา
2. การกำหนด **Scope Strategy** (กว้าง vs เจาะจง vs Hybrid Pattern-Aware)
3. การวิเคราะห์ **Case Study: Large Weather Time-Series Training** และแนวทางการแก้ปัญหา

---

## 🗺️ 1. Taxonomy of Auto-Tuning in ML Data Pipeline

การ Auto-tune ระบบ Data Loading สำหรับ Machine Learning สามารถจำแนกตาม **5 มิติหลัก (5 Dimensions)** โดยรวบรวมเปเปอร์จากทั้ง **Layer 1 (เปเปอร์หลัก)** และ **Layer 2 (งานวิจัยที่อ้างอิง/รากฐาน เช่น Pecan, Cachew, FastFlow, PreSto, FusionFlow, DALI, FFCV)** รวมทั้งสิ้น **20 เปเปอร์ + งานวิจัยของเรา** ดังนี้:

### 📊 ตารางที่ 1: การจำแนกตามมิติและหมวดหมู่อย่างละเอียด (Sub-Category Mapping - All Layers)

| มิติ (Dimension)                                           | หมวดหมู่ย่อย (Sub-Category)                                                                                                                                                                                                                                                                                                         | เปเปอร์ที่จัดอยู่ในหมวดนี้ (Paper Mapping - Layer 1 & 2)                                                                                                                                                                                                                            |
| :--------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Tuning Target**<br>*(เป้าหมายการจูน)*                 | • **Parallel Workers & Placement:** ปรับจำนวน worker/ตำแหน่ง<br>• **Buffer / Prefetch Factor:** ปรับคิว/ขนาดบัฟเฟอร์ความจำ<br>• **Caching Tier & Reuse:** ปรับการแคชข้ามระดับ RAM/Storage<br>• **Transform Order & Kernels:** สลับลำดับการแปลงข้อมูล/CUDA<br>• **Batch Size & Cluster Resources:** ปรับ Batch Size/GPUs             | ➔ PyTorch, tf.data, Synergy, MinatoLoader, Pecan, FastFlow<br>➔ tf.data, DLCache, **งานของเรา ⭐**<br>➔ CoorDL, DLCache, Cachew, Joader<br>➔ Pecan (`AutoOrder`), NVIDIA DALI (CUDA Kernels)<br>➔ Pollux *(โฟกัส Cluster-level scheduling)*                                          |
| **2. Control Mechanism**<br>*(กลไกการควบคุม)*              | • **Static / Manual / Hardcoded Rules:** ใช้ค่าคงที่/สคริปต์<br>• **Heuristics / Profiling Threshold:** วัดเวลา warmup (P75/P90)<br>• **Reactive Feedback Loop:** ปรับตาม Throughput ย้อนหลัง<br>• **Mathematical Optimization Model:** Queueing Theory, LP, $\omega$<br>• **Proactive Memory-Aware & Rollback:** เช็ก RAM ก่อน OOM | ➔ PyTorch Standard, ConcurrentDataLoader, FFCV, SPDL<br>➔ MinatoLoader (P75 Timeout), SpeedyLoader<br>➔ tf.data (AUTOTUNE Hill-Climbing), Cachew<br>➔ Plumber (LP), DLCache ($\omega$ rate), Synergy (Queueing)<br>➔ **งานของเรา (Proactive Memory Profiling + Safety Rollback) ⭐** |
| **3. Architecture & Offloading**<br>*(สถาปัตยกรรมและโหนด)* | • **Local In-Node Loading:** รันประมวลผลบน GPU Node เดียวกัน<br>• **Remote CPU / Disaggregated Offloading:** ย้ายงานไป CPU Server<br>• **Distributed Coordinated Cache:** แชร์ Cache ข้ามหลายโหนด<br>• **Specialized Accelerator / Hardware:** ใช้ In-Storage / FPGA                                                                | ➔ PyTorch, tf.data, MinatoLoader, Synergy, FFCV, SPDL, **งานของเรา ⭐**<br>➔ DLCache (ZeroMQ/NFS), Cedar, Pecan, Cachew, FastFlow<br>➔ CoorDL (MinIO Cache), Joader<br>➔ NVIDIA DALI (GPU), PreSto (In-Storage), FusionFlow (Hybrid)                                                 |
| **4. Hardware Environment**<br>*(สภาพแวดล้อมฮาร์ดแวร์)*    | • **HPC / High-RAM Cluster:** RAM 512GB+, Remote Storage/NFS<br>• **Commodity Hardware:** RAM จำกัด (16-32GB), Local NVMe/SSD<br>• **Specialized Hardware / In-Storage:** ใช้ Custom HW                                                                                                                                             | ➔ FFCV, NVIDIA DALI, MinatoLoader, DLCache, Pecan, FastFlow<br>➔ PyTorch standard, ConcurrentDataLoader, **งานของเรา ⭐**<br>➔ PreSto (In-Storage NVMe/FPGA), FusionFlow                                                                                                             |
| **5. Data Access Pattern**<br>*(รูปแบบการเข้าถึงข้อมูล)*   | • **Independent Samples:** รูปภาพเดี่ยว (ImageNet, PIL read)<br>• **Large Grid / Tensor Data:** Weather, Space-Time (NetCDF/HDF5)<br>• **Variable Prep-Time Samples:** เวลาประมวลผลแต่ละรูปไม่เท่ากัน<br>• **Recommendation / Sparse Data:** ข้อมูล Recommendation System                                                           | ➔ PyTorch DataLoader, FFCV, tf.data, Cachew<br>➔ Xarray, Zarr Loaders, **เคส Weather Time-Series ⭐**<br>➔ MinatoLoader (KiTS19 3D Medical Images)<br>➔ PreSto, Joader                                                                                                               |

---

### 📋 ตารางที่ 2: สรุปกางแยกรายเปเปอร์ครบทั้ง 2 Layer (Paper-by-Paper Classification Matrix)

ตารางนี้กางเปเปอร์รวมทั้งสิ้น **20 เปเปอร์ + งานวิจัยของเรา** เทียบกับมิติทั้ง 5 อย่างสมบูรณ์:

| ชื่อเปเปอร์ / ระบบ              | Layer    | 1. Tuning Target                                          | 2. Control Mechanism                  | 3. Architecture                | 4. Hardware Target          | 5. Primary Workload                         |
| :------------------------------ | :------- | :-------------------------------------------------------- | :------------------------------------ | :----------------------------- | :-------------------------- | :------------------------------------------ |
| **PyTorch DataLoader** `[9]`    | Baseline | Manual (`num_workers`)                                    | Static / Manual                       | Local In-Node                  | Commodity & HPC             | General / Images                            |
| **tf.data (Google)** `[35]`     | L1       | Thread / Buffer                                           | Reactive Feedback (Hill-Climbing)     | Local In-Node (C++)            | Commodity & HPC             | General Data Pipelines                      |
| **Plumber (2022)**              | L1       | Workers / Memory                                          | Math (Linear Programming)             | Local In-Node                  | HPC Server                  | General Pipelines                           |
| **CoorDL (2021)** `[34]`        | L1/L2    | Cache Coordination                                        | Coordinated Caching                   | Distributed Shared Cache       | HPC Cluster                 | Shared Multi-Node Image                     |
| **Pollux (2021)**               | L1       | Batch Size / GPUs                                         | Math (Goodput Co-adaptation)          | Cluster Scheduler              | Cloud Cluster               | Distributed Training                        |
| **MinatoLoader (2026)**         | L1       | Workers / Fast-Slow Queue                                 | Heuristic (P75 Profiling)             | Local In-Node                  | HPC (512GB RAM)             | Variable Prep-Time (3D Medical)             |
| **DLCache (2023)**              | L1       | Workers / Eviction Score                                  | Math ($\omega = t_{fetch}/t_{req}$)   | Remote CPU Offloading (ZeroMQ) | Remote NFS Storage          | Large Distributed Datasets                  |
| **ConcurrentDataLoader**        | L1       | Async Concurrency                                         | Static Async/Thread Pool              | Local In-Node                  | Commodity HW                | Web / Network IO-Bound                      |
| **Synergy (2024)**              | L1       | Workers                                                   | Math (Queueing Theory)                | Local In-Node                  | Single GPU Node             | General Pipelines                           |
| **Cedar (2024)**                | L1       | Pipeline Graph                                            | Static Graph Optimization             | Remote CPU Offloading          | Cloud Cluster               | Multi-System Pipelines                      |
| **Pecan (ATC '24)** `[18]`      | L2       | Workers (`AutoPlacement`) + Transform Order (`AutoOrder`) | Static Rules + Pipeline Reordering    | Remote CPU / Disaggregated     | Distributed Cluster         | TensorFlow Pipelines                        |
| **Cachew (ATC '22)** `[17]`     | L2       | Remote Workers / Cache                                    | Cloud Service Auto-Scaling            | Cloud Shared Cache Service     | Cloud Infrastructure        | Cloud ML Pipelines                          |
| **FastFlow (VLDB '23)** `[45]`  | L2       | Offload Pipeline Scaling                                  | Smart Remote CPU Offloading           | Remote CPU Offloading          | Disaggregated Cluster       | Heavy Preprocessing                         |
| **NVIDIA DALI** `[4]`           | L2       | GPU CUDA Kernels                                          | Accelerator Offloading                | GPU Accelerator Offloading     | Dedicated GPU HW            | Heavy Image/Video Augments                  |
| **SpeedyLoader ('24)** `[37]`   | L2       | Pipeline Profiling                                        | Static Profiling Analysis             | Local In-Node                  | Single Server               | PyTorch Bottleneck Study                    |
| **PreSto (ISCA '24)** `[25]`    | L2       | In-Storage Processing                                     | Specialized Hardware Acceleration     | Specialized In-Storage HW      | Custom NVMe/FPGA            | Recommendation Systems                      |
| **FusionFlow (VLDB '23)**`[21]` | L2       | Hybrid CPU-GPU Prep                                       | Hybrid Pipeline Offloading            | Accelerator/CPU Hybrid         | Hybrid CPU-GPU Server       | Heavy Preprocessing                         |
| **FFCV (2022)**                 | L2       | Custom `.ffcv` Format                                     | High-Speed C++ Mmap                   | Local In-Node                  | HPC Server / NVMe           | Computer Vision Datasets                    |
| **SPDL (2026)**                 | L2       | Free-threading (Py 3.13)                                  | Multi-threading GIL-free              | Local In-Node                  | Commodity & HPC             | Media Processing Pipelines                  |
| **Joader (2023)**               | L2       | Cross-job Data Fetching                                   | Data-Centric Shared Fetching          | Distributed Shared Cache       | Multi-job Cluster           | Multi-model Training Jobs                   |
| 👑 **งานวิจัยของเรา**           | Target   | `num_workers` + `prefetch` + Sliding Buffer               | **Proactive Memory-Aware + Rollback** | **Local In-Node Middleware**   | **Commodity (16-32GB RAM)** | **Pattern-Aware (Weather Tensor & Images)** |

---

## 🎯 2. Scope Strategy: การเลือกทิศทางงานวิจัย

อาจารย์ตั้งโจทย์ 3 ทางเลือกสำหรับการดำเนินงาน:

```
[ทางเลือก 1: งานกว้าง]  ---> ครอบคลุมทุก Use Case (Image, Text, Tabular)
[ทางเลือก 2: งานเจาะจง] ---> เน้นเฉพาะ Problem เช่น Weather Time-Series
[ทางเลือก 3: Hybrid]    ---> Pattern-Aware Adaptive Middleware (แนะนำ! ⭐)
```

### เปรียบเทียบ 3 ทางเลือก:

1. **งานกว้าง (General-Purpose Middleware):**
   - _ข้อดี:_ Impact กว้าง ผู้ใช้ PyTorch ทั่วไปใช้ได้ทันที
   - _ข้อเสีย:_ ยากที่จะให้ Auto-tuner ตัวเดียวฉลาดกับทุกโครงสร้างข้อมูลโดยไม่รู้พฤติกรรมข้อมูล
2. **งานเจาะจง (Domain-Specific เช่น Weather Time-Series):**
   - _ข้อดี:_ เห็นปัญหา I/O และ RAM Bottleneck ชัดเจนมาก (เคสไฟล์ใหญ่ NetCDF/HDF5)
   - _ข้อเสีย:_ Scope อาจดูแคบเกินไปสำหรับวิทยานิพนธ์ระดับมหาลัย หากไม่รองรับ Workload อื่น
3. **Hybrid / Pattern-Aware Approach (แนวทางที่ดีที่สุด ✨):**
   - ออกแบบเป็น **Pattern-Aware Auto-Tuning Middleware** ที่จำแนกและปรับ Strategy อัตโนมัติในระดับ Runtime:
     - 🟢 **Pattern A (Fit-in-RAM):** ถ้าระบบตรวจพบ Dataset Size < RAM Budget $\rightarrow$ สลับเป็น **Auto In-Memory Caching** (โหลดครั้งเดียว ไม่มี I/O ระหว่าง Epoch)
     - 🟡 **Pattern B (Large Tensor Stream / Weather Case):** ถ้า Dataset > RAM & อ่านไฟล์ใหญ่ $\rightarrow$ สลับเป็น **Smart Chunk Prefetching + Parallel Worker Async IO**
     - 🔵 **Pattern C (CPU Heavy Augmentation):** ถ้า CPU Preprocessing ช้ากว่า I/O $\rightarrow$ ปรับสเกล `num_workers` ชิดขอบเขต RAM/CPU Cores

---

## 🌧️ 3. Case Study Analysis: Weather Time-Series Training

### วิเคราะห์ปัญหาในเคสของนิสิตอาจารย์:

ในงานวิจัย Weather / Climate Forecasting ข้อมูลมักอยู่ในรูปแบบ Multi-dimensional Tensor ขนาดใหญ่ (เช่นไฟล์ NetCDF4 / HDF5 / Zarr หลายร้อย GB):

```
[แนวทางเดิม 1: Load All to RAM]
- Cold Start Overhead สูงมาก (เสียเวลารอหลายสิบนาที/เป็นชั่วโมงก่อนเริ่ม Epoch 1)
- ต้องใช้ HPC Server RAM 256-512GB (บน Commodity Hardware 16-32GB เกิด OOM ล่มทันที)

[แนวทางเดิม 2: Load File-by-File]
- เกิด I/O Bottleneck หนักมาก (File Handle Opening Overhead, Disk Seek, HDF5 Decompression)
- GPU Starvation เกิดขึ้นตลอดเวลาเนื่องจากไม่มี Auto-tuning/Prefetching ที่เหมาะสม
```

### 💡 โซลูชันจาก Middleware ของเราสำหรับเคสนี้:

```
                          ┌─────────────────────────────────────────┐
                          │ Proactive Memory-Aware Auto-Tuner        │
                          └───────────────────┬─────────────────────┘
                                              │
               ┌──────────────────────────────┴──────────────────────────────┐
               ▼                                                             ▼
  [ Hardware Profile ]                                      [ Data Access Pattern ]
  • Free RAM: 16 GB                                         • File Format: NetCDF/HDF5
  • Disk Speed: NVMe SSD (2 GB/s)                           • Dataset Size: 120 GB (> RAM)
               │                                                             │
               └──────────────────────────────┬──────────────────────────────┘
                                              ▼
                    ┌───────────────────────────────────────────────────┐
                    │      Auto-Configured Sliding Buffer Pipeline      │
                    │                                                   │
                    │  1. Prefetch Window Size = 10 GB (Safe inside RAM)│
                    │  2. Worker Threads = 6 (Async I/O Overlapping)   │
                    │  3. Chunking = Read contiguous spatial slices     │
                    └─────────────────────────┬─────────────────────────┘
                                              ▼
                              [ GPU Training Zero Starvation ]
```

1. **Sliding Prefetch Buffer:** แทนที่จะอัดเข้า RAM ทั้งหมด หรือ โหลดทีละไฟล์ย่อย Middleware จะสร้าง **Prefetch Buffer ที่จำกัดขนาดตาม RAM ที่ปลอดภัย** (เช่น ใช้ RAM ไม่เกิน 70% ของเครื่อง)
2. **Async Overlapping:** ในขณะที่ GPU กำลังเทรน Batch ปัจจุบัน Multi-worker จะอ่านและ Deserialize Chunk ของ Time-Series ถัดไปรอไว้ใน Shared Memory ล่วงหน้า
3. **Zero Cold-Start:** เริ่มเทรนได้ทันทีที่ Chunk แรกพร้อม ไม่ต้องรอโหลดทั้ง Dataset 120GB เข้า RAM
4. **Safety Rollback & Dynamic Adjustment:** ถ้า I/O ช้าลง หรือ RAM เริ่มตึง ระบบจะปรับลด Prefetch Factor ลงอัตโนมัติ ไม่ให้เกิด OOM

---

## 📌 อ้างอิงไฟล์ที่เกี่ยวข้อง (Related Links)

- [[01_Research_Idea_Refined]] — โครงร่างแผนวิทยานิพนธ์ฉบับปรับปรุง
- [[/01_Knowledge_Base/02_Notes/MinatoLoader_Citation_and_Foundation_Analysis.md|MinatoLoader Analysis]] — การวิเคราะห์เปเปอร์ SOTA MinatoLoader
- [[/01_Knowledge_Base/02_Notes/Data_Pipeline_Bottlenecks_and_PyTorch_Limitations.md|Data Pipeline Bottlenecks]] — สรุปข้อจำกัด PyTorch DataLoader และ SOTA Papers
