---
tags:
  - taxonomy
  - research-idea
  - advisor-feedback
  - auto-tuning
aliases:
  - AutoTuning_Taxonomy
---

# 📐 Taxonomy of ML Data Loading Auto-Tuning & Citation Analysis

_(ปรับปรุงล่าสุด - กันยายน 2026)_

เอกสารนี้จัดทำขึ้นเพื่อรวบรวม **Taxonomy ของระบบ Auto-Tuning ใน ML Data Pipeline** ทั้งหมดที่เกี่ยวข้องกับงานวิจัยของเรา โดยจัดหมวดหมู่ระบบและงานวิจัยที่อ้างอิง (Citations) จากเปเปอร์หลัก **MinatoLoader (EuroSys '26)** และเปเปอร์ในสายตระกูลใกล้เคียง:

1. **Taxonomy 5 มิติ** ของระบบ Auto-tuning
2. การระบุตำแหน่งงานวิจัยของเรา (Where We Stand)
3. การกำหนด **Scope Strategy** (กว้าง vs เจาะจง vs Hybrid Pattern-Aware)
4. การวิเคราะห์ **Case Study: Large Weather Time-Series Training**

---

## 🗺️ 1. Taxonomy of Auto-Tuning in ML Data Pipeline

การ Auto-tune ระบบ Data Loading สำหรับ Machine Learning จำแนกตาม **5 มิติหลัก (5 Dimensions)** โดยรวบรวมระบบจากสายการอ้างอิง (Citation Lineage):

- **Baseline & SOTA Frameworks:** PyTorch DataLoader, tf.data, MinatoLoader (EuroSys '26)
- **MinatoLoader Direct Citations (Layer 1 & 2):** Pecan (ATC '24), Cachew (ATC '22), FastFlow (VLDB '23), NVIDIA DALI, SpeedyLoader (MLSys '24), PreSto (ISCA '24), FusionFlow (VLDB '23), CoorDL (VLDB '21), Plumber, DLCache, ConcurrentDataLoader, Pollux
- **Foundational Systems (Systems Lineage):** SEDA (Stage+Queue), Optimus (Data Skew), DS2 (Single-step Rate Matching), Autopilot (Google EuroSys '20), CherryPick, Quasar, Hellerstein/Deshpande (AutoOrder roots)
- **SOTA General Extensions:** Cedar, FFCV, SPDL, Joader _(ขยายความครอบคลุมระบบ Data Pipeline)_

---

### 📊 ตารางที่ 1: การจำแนกตามมิติและหมวดหมู่อย่างละเอียด (Sub-Category Mapping)

| มิติ (Dimension)                                           | หมวดหมู่ย่อย (Sub-Category)                                                                                                                                                                                                                                                                                                              | เปเปอร์และระบบที่จัดอยู่ในหมวดนี้ (Paper Mapping)                                                                                                                                                                                                                                                                                                 |
| :--------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **1. Tuning Target**<br>_(เป้าหมายการจูน)_                 | • **Parallel Workers & Placement:** ปรับจำนวน worker/ตำแหน่ง<br>• **Buffer / Prefetch Factor:** ปรับคิว/ขนาดบัฟเฟอร์ความจำ<br>• **Caching Tier & Reuse:** ปรับการแคชข้ามระดับ RAM/Storage<br>• **Transform Order & Kernels:** สลับลำดับการแปลงข้อมูล/CUDA<br>• **Batch Size & Cluster Resources:** ปรับ Batch Size/GPUs/VMs              | ➔ PyTorch, tf.data, MinatoLoader, Pecan, FastFlow, DS2, SEDA<br>➔ tf.data, DLCache, Plumber, **งานของเรา ⭐**<br>➔ CoorDL, DLCache, Cachew, Joader, Auto-caching (Agrawal)<br>➔ Pecan (`AutoOrder`), DALI (CUDA Kernels), Hellerstein/Deshpande<br>➔ Pollux, CherryPick (VM sizing), Quasar, Autopilot (Google)                                   |
| **2. Control Mechanism**<br>_(กลไกการควบคุม)_              | • **Static / Manual / Hardcoded Rules:** ใช้ค่าคงที่/สคริปต์<br>• **Heuristics / Profiling Threshold:** วัดเวลา warmup (P75/P90)<br>• **Reactive Feedback Loop:** ปรับตาม Throughput ย้อนหลัง<br>• **Mathematical Optimization Model:** Queueing Theory, LP, $\omega$, DS2<br>• **Proactive Memory-Aware & Rollback:** เช็ก RAM ก่อน OOM | ➔ PyTorch Standard, ConcurrentDataLoader, FFCV, SPDL<br>➔ MinatoLoader (P75 Timeout), SpeedyLoader, Optimus (Data Skew)<br>➔ tf.data (AUTOTUNE Gradient Descent + M/M/1/k Model), Cachew, Quasar<br>➔ Plumber (LP), DLCache ($\omega$ rate), DS2 (Single-step Rate Matching)<br>➔ **งานของเรา (Proactive Memory Profiling + Safety Rollback) ⭐** |
| **3. Architecture & Offloading**<br>_(สถาปัตยกรรมและโหนด)_ | • **Local In-Node Loading:** รันประมวลผลบน GPU Node เดียวกัน<br>• **Remote CPU / Disaggregated Offloading:** ย้ายงานไป CPU Server<br>• **Distributed Coordinated Cache:** แชร์ Cache ข้ามหลายโหนด<br>• **Specialized Accelerator / Hardware:** ใช้ In-Storage / FPGA                                                                     | ➔ PyTorch, tf.data, MinatoLoader, FFCV, SPDL, SEDA, **งานของเรา ⭐**<br>➔ DLCache (ZeroMQ/NFS), Cedar, Pecan, Cachew, FastFlow, Autopilot<br>➔ CoorDL (MinIO Cache), Joader<br>➔ NVIDIA DALI (GPU), PreSto (In-Storage), FusionFlow (Hybrid)                                                                                                      |
| **4. Hardware Environment**<br>_(สภาพแวดล้อมฮาร์ดแวร์)_    | • **HPC / High-RAM Cluster:** RAM 512GB+, Remote Storage/NFS<br>• **Commodity Hardware:** RAM จำกัด (16-32GB), Local NVMe/SSD<br>• **Specialized Hardware / In-Storage:** ใช้ Custom HW                                                                                                                                                  | ➔ FFCV, NVIDIA DALI, MinatoLoader, DLCache, Pecan, FastFlow, Quasar<br>➔ PyTorch standard, ConcurrentDataLoader, **งานของเรา ⭐**<br>➔ PreSto (In-Storage NVMe/FPGA), FusionFlow                                                                                                                                                                  |
| **5. Data Access Pattern**<br>_(รูปแบบการเข้าถึงข้อมูล)_   | • **Independent Samples:** รูปภาพเดี่ยว (ImageNet, PIL read)<br>• **Large Grid / Tensor Data:** Weather, Space-Time (NetCDF/HDF5)<br>• **Variable Prep-Time / Skewed Data:** เวลาอ่านแต่ละตัวไม่เท่ากัน<br>• **Recommendation / Sparse Data:** ข้อมูล Recommendation System                                                              | ➔ PyTorch DataLoader, FFCV, tf.data, Cachew<br>➔ Xarray, Zarr Loaders, **เคส Weather Time-Series ⭐**<br>➔ MinatoLoader (3D Medical Images), Optimus (Data Skew)<br>➔ PreSto, Joader                                                                                                                                                              |

---

### 📋 ตารางที่ 2: สรุปกางแยกรายเปเปอร์ครบทุกระบบ (Paper Classification Matrix)

ตารางกางเปเปอร์และระบบรวมทั้งสิ้น **22 รายการ** พร้อมลิงก์ไฟล์ PDF ที่ดาวน์โหลดเก็บในเครื่อง (`01_Knowledge_Base/01_Papers/`):

| ชื่อเปเปอร์ / ระบบ              | Citation Origin (Cited by)                                                                                     | ไฟล์ PDF ในเครื่อง                                                                                                                                                                                                | 1. Tuning Target                                    | 2. Control Mechanism                           | 3. Architecture                  | 4. Hardware Target                                | 5. Primary Workload                         |
| :------------------------------ | :------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------- | :--------------------------------------------- | :------------------------------- | :------------------------------------------------ | :------------------------------------------ |
| **PyTorch DataLoader** `[9]`    | PyTorch Baseline                                                                                               | 📄 _Docs_                                                                                                                                                                                                         | Manual (`num_workers`)                              | Static / Manual                                | Local In-Node                    | Commodity & HPC                                   | General / Images                            |
| **tf.data (Google)** `[35]`     | Cited by FastFlow, Lotus, SpeedyLoader, Pecan, FFCV, FusionFlow, PreSto, Cachew, MinatoLoader, Plumber, Cedar  | 📄 [tf_data.pdf](file:///home/mew/Desktop/mew/study/Master%20degree/thesis/01_Knowledge_Base/01_Papers/tf_data.pdf)                                                                                               | Thread / Buffer                                     | Reactive Feedback (Gradient Descent + M/M/1/k) | Local In-Node (C++)              | Commodity & HPC                                   | General Data Pipelines                      |
| **Plumber (2022)**              | Cited by FastFlow, Lotus, Pecan, PreSto, Cachew, Cedar                                                         | 📄 [Plumber.pdf](file:///home/mew/Desktop/mew/study/Master%20degree/thesis/01_Knowledge_Base/01_Papers/Plumber.pdf)                                                                                               | Workers / Memory                                    | Math (Linear Programming)                      | Local In-Node                    | HPC Server                                        | General Pipelines                           |
| **CoorDL (2021)** `[34]`        | Cited by tf.data, Cachew, MinatoLoader, Joader, Cedar                                                          | 📄 [CoorDL.pdf](file:///home/mew/Desktop/mew/study/Master%20degree/thesis/01_Knowledge_Base/01_Papers/CoorDL.pdf)                                                                                                 | Cache Coordination                                  | Coordinated Caching                            | Distributed Shared Cache         | HPC Cluster                                       | Shared Multi-Node Image                     |
| **Pollux (2021)**               | Cited by MinatoLoader `[22]`                                                                                   | 📄 [Pollux.pdf](file:///home/mew/Desktop/mew/study/Master%20degree/thesis/01_Knowledge_Base/01_Papers/Pollux.pdf)                                                                                                 | Batch Size / GPUs                                   | Math (Goodput Co-adaptation)                   | Cluster Scheduler                | Cloud Cluster                                     | Distributed Training                        |
| **MinatoLoader (2026)**         | SOTA Target Baseline                                                                                           | 📄 [MinatoLoader.pdf](file:///home/mew/Desktop/mew/study/Master%20degree/thesis/01_Knowledge_Base/01_Papers/MinatoLoader.pdf)                                                                                     | Workers / Fast-Slow Queue                           | Heuristic (P75 Profiling)                      | Local In-Node                    | HPC (512GB RAM)                                   | Variable Prep-Time (Medical)                |
| **DLCache (2023)**              | Cited by MinatoLoader `[23]`                                                                                   | 📄 [DLCache.pdf](file:///home/mew/Desktop/mew/study/Master%20degree/thesis/01_Knowledge_Base/01_Papers/DLCache.pdf)                                                                                               | Workers / Eviction Score                            | Math ($\omega = t_{fetch}/t_{req}$)            | Remote CPU Offloading (ZeroMQ)   | Remote NFS Storage                                | Large Distributed Datasets                  |
| **ConcurrentDataLoader**        | Concurrency Baseline                                                                                           | 📄 [ConcurrentDataLoader.pdf](file:///home/mew/Desktop/mew/study/Master%20degree/thesis/01_Knowledge_Base/01_Papers/ConcurrentDataLoader.pdf)                                                                     | Async Concurrency                                   | Static Async/Thread Pool                       | Local In-Node                    | Commodity HW                                      | Web / Network IO-Bound                      |
| **Pecan (ATC '24)** `[18]`      | Cited by Lotus, MinatoLoader                                                                                   | 📄 [Pecan.pdf](file:///home/mew/Desktop/mew/study/Master%20degree/thesis/01_Knowledge_Base/01_Papers/Pecan.pdf)                                                                                                   | Workers (`AutoPlacement`) + Transform (`AutoOrder`) | Static Rules + Pipeline Reorder                | Remote CPU / Disaggregated       | Distributed Cluster                               | TensorFlow Pipelines                        |
| **Cachew (ATC '22)** `[17]`     | Cited by FastFlow, Lotus, SpeedyLoader, Pecan, FusionFlow, PreSto, MinatoLoader, Cedar, DLCache                | 📄 [Cachew.pdf](file:///home/mew/Desktop/mew/study/Master%20degree/thesis/01_Knowledge_Base/01_Papers/Cachew.pdf)                                                                                                 | Remote Workers / Cache                              | Cloud Service Auto-Scaling                     | Cloud Shared Cache Service       | Cloud Infrastructure                              | Cloud ML Pipelines                          |
| **FastFlow (VLDB '23)** `[45]`  | Cited by Lotus, Pecan, FusionFlow, PreSto, MinatoLoader, Cedar                                                 | 📄 [FastFlow.pdf](file:///home/mew/Desktop/mew/study/Master%20degree/thesis/01_Knowledge_Base/01_Papers/FastFlow.pdf)                                                                                             | Offload Pipeline Scaling                            | Smart Remote CPU Offloading                    | Remote CPU Offloading            | Disaggregated Cluster                             | Heavy Preprocessing                         |
| **NVIDIA DALI** `[4]`           | Cited by tf.data, FastFlow, Lotus, Pecan, FFCV, FusionFlow, PreSto, Cachew, MinatoLoader, Plumber, Cedar, SPDL | 📄 [NVIDIA_DALI.pdf](file:///home/mew/Desktop/mew/study/Master%20degree/thesis/01_Knowledge_Base/01_Papers/NVIDIA_DALI.pdf) _(หมายเหตุ: เป็น Huawei benchmark study เกี่ยวกับ DALI ไม่ใช่ official NVIDIA paper)_ | GPU CUDA Kernels                                    | Accelerator Offloading                         | GPU Accelerator Offloading       | Dedicated GPU HW                                  | Heavy Image/Video Augments                  |
| **SpeedyLoader ('24)** `[37]`   | Cited by Lotus, MinatoLoader                                                                                   | 📄 [SpeedyLoader.pdf](file:///home/mew/Desktop/mew/study/Master%20degree/thesis/01_Knowledge_Base/01_Papers/SpeedyLoader.pdf)                                                                                     | Pipeline Profiling                                  | Static Profiling Analysis                      | Local In-Node                    | Single Server                                     | PyTorch Bottleneck Study                    |
| **PreSto (ISCA '24)** `[25]`    | Cited by FastFlow, MinatoLoader, Cedar                                                                         | 📄 [PreSto.pdf](file:///home/mew/Desktop/mew/study/Master%20degree/thesis/01_Knowledge_Base/01_Papers/PreSto.pdf)                                                                                                 | In-Storage Processing                               | Specialized Hardware Accel                     | Specialized In-Storage HW        | Custom NVMe/FPGA                                  | Recommendation Systems                      |
| **FusionFlow (VLDB '23)**`[21]` | Cited by MinatoLoader                                                                                          | 📄 [FusionFlow.pdf](file:///home/mew/Desktop/mew/study/Master%20degree/thesis/01_Knowledge_Base/01_Papers/FusionFlow.pdf)                                                                                         | Hybrid CPU-GPU Prep                                 | Hybrid Pipeline Offloading                     | Accelerator/CPU Hybrid           | Hybrid CPU-GPU Server                             | Heavy Preprocessing                         |
| **SEDA (SOSP '01)**             | Cited by tf.data                                                                                               | 📄 [SEDA.pdf](file:///home/mew/Desktop/mew/study/Master%20degree/thesis/01_Knowledge_Base/01_Papers/SEDA.pdf)                                                                                                     | Thread Pool Sizing                                  | Staged Queues + Dynamic Controller             | Staged Event-Driven Architecture | General Server                                    | Multi-stage Pipelines                       |
| **Optimus Framework**           | Cited by tf.data, Pollux, Plumber                                                                              | 📄 [Optimus.pdf](file:///home/mew/Desktop/mew/study/Master%20degree/thesis/01_Knowledge_Base/01_Papers/Optimus.pdf)                                                                                               | Worker Data Partitioning                            | Dynamic EPG Rewriting (Data Skew)              | Worker Splitting / Re-partition  | Distributed Systems (General, ไม่ใช่ ML-specific) | Skewed Data-Parallel Processing             |
| **DS2 (OSDI '18)**              | Cited by tf.data                                                                                               | 📄 [DS2.pdf](file:///home/mew/Desktop/mew/study/Master%20degree/thesis/01_Knowledge_Base/01_Papers/DS2.pdf)                                                                                                       | Worker Sizing                                       | Single-step Rate Calculation                   | Streaming Data Engine            | Distributed Stream                                | Real-time Streaming                         |
| **Autopilot (EuroSys '20)**     | Cited by Pecan                                                                                                 | 📄 [Autopilot.pdf](file:///home/mew/Desktop/mew/study/Master%20degree/thesis/01_Knowledge_Base/01_Papers/Autopilot.pdf)                                                                                           | CPU/RAM & Parallelism                               | Dynamic Resource Manager                       | Cluster Autoscaler               | Google Infrastructure                             | Cloud Microservices & ML                    |
| **CherryPick (NSDI '17)**       | Cited by Cachew, Autopilot                                                                                     | 📄 [CherryPick.pdf](file:///home/mew/Desktop/mew/study/Master%20degree/thesis/01_Knowledge_Base/01_Papers/CherryPick.pdf)                                                                                         | Cloud VM Configuration                              | Bayesian Optimization                          | Cloud Resource Profiler          | Public Cloud (AWS)                                | Big Data Analytics                          |
| **Quasar (ASPLOS '14)**         | Cited by Cachew, Autopilot, CherryPick                                                                         | 📄 [Quasar.pdf](file:///home/mew/Desktop/mew/study/Master%20degree/thesis/01_Knowledge_Base/01_Papers/Quasar.pdf)                                                                                                 | Resource Allocation                                 | Performance-based Profiling                    | Cluster Resource Manager         | Datacenter                                        | Multi-tenant Cluster                        |
| **Cedar (2024)**                | DL Data Loader SOTA                                                                                            | 📄 [Cedar.pdf](file:///home/mew/Desktop/mew/study/Master%20degree/thesis/01_Knowledge_Base/01_Papers/Cedar.pdf)                                                                                                   | Pipeline Graph                                      | Static Graph Optimization                      | Remote CPU Offloading            | Cloud Cluster                                     | Multi-System Pipelines                      |
| **FFCV (2022)**                 | DL Data Loader SOTA                                                                                            | 📄 [FFCV.pdf](file:///home/mew/Desktop/mew/study/Master%20degree/thesis/01_Knowledge_Base/01_Papers/FFCV.pdf)                                                                                                     | Custom `.ffcv` Format                               | High-Speed C++ Mmap                            | Local In-Node                    | HPC Server / NVMe                                 | Computer Vision Datasets                    |
| **SPDL (2026)**                 | DL Data Loader SOTA                                                                                            | 📄 [SPDL.pdf](file:///home/mew/Desktop/mew/study/Master%20degree/thesis/01_Knowledge_Base/01_Papers/SPDL.pdf)                                                                                                     | Free-threading (Py 3.13)                            | Multi-threading GIL-free                       | Local In-Node                    | Commodity & HPC                                   | Media Processing Pipelines                  |
| **Joader (2023)**               | DL Data Loader SOTA                                                                                            | 📄 [Joader.pdf](file:///home/mew/Desktop/mew/study/Master%20degree/thesis/01_Knowledge_Base/01_Papers/Joader.pdf)                                                                                                 | Cross-job Data Fetching                             | Data-Centric Shared Fetching                   | Distributed Shared Cache         | Multi-job Cluster                                 | Multi-model Training Jobs                   |
| 👑 **งานวิจัยของเรา**           | Target                                                                                                         | 🌟 _Our Thesis_                                                                                                                                                                                                   | `num_workers` + `prefetch` + Sliding Buffer         | **Proactive Memory-Aware + Rollback**          | **Local In-Node Middleware**     | **Commodity (16-32GB RAM)**                       | **Pattern-Aware (Weather Tensor & Images)** |

---

### 📊 เปเปอประเภท Survey & Empirical Bottleneck Studies (ดาวน์โหลดในเครื่องครบแล้ว)

- 📄 **[Empirical_Study_Low_GPU_Utilization.pdf](file:///home/mew/Desktop/mew/study/Master%20degree/thesis/01_Knowledge_Base/01_Papers/Empirical_Study_Low_GPU_Utilization.pdf)** _(Cited in MinatoLoader)_  
  • **โฟกัส:** วิเคราะห์หาสาเหตุที่ GPU ว่างงาน (GPU Idle Stalls) $\rightarrow$ สรุปว่าคอขวดอันดับ 1 เกิดจาก Data Preprocessing & Ingestion _(ใช้อ้างอิง Problem Statement ในบทที่ 1)_
- 📄 **[Lotus.pdf](file:///home/mew/Desktop/mew/study/Master%20degree/thesis/01_Knowledge_Base/01_Papers/Lotus.pdf)** _(Cited in MinatoLoader)_  
  • **โฟกัส:** วิเคราะห์คอขวดของการเทรน ML ทั้งฝั่ง Framework (PyTorch DataLoader, NVIDIA DALI, tf.data) และฝั่ง Hardware (CPU Cores, Memory Floor, Disk I/O) _(ใช้อ้างอิง Background ในบทที่ 2)_

---

### 📍 สรุปจุดยืนงานวิจัยของเราใน 5 มิติ (Where We Stand)

> **"Local In-Node, Proactive Memory-Aware Auto-Tuning Middleware on Commodity Hardware with Pattern-Aware Adaptation"**

1. **Architecture:** **Local In-Node** (สวมทับ DataLoader เดิมใน Node เดียวกัน ไม่ต้องตั้ง Server หรือย้ายไป CPU Cluster อื่นแบบ DLCache/Cachew)
2. **Control Mechanism:** **Proactive Memory Profiling** (ประเมิน RAM ก่อนสั่งสร้าง Worker ป้องกัน OOM) + **Safety Rollback** (ถ้าย้อนหลังแล้ว Throughput ตก ให้ Revert ทันที)
3. **Hardware Environment:** โฟกัส **Commodity Hardware** (RAM 16-32GB, Single-node SSD)
4. **Tuning Target:** `num_workers` + `prefetch_factor` + Sliding Memory Buffer (ไม่ปรับ Batch Size เพื่อรักษา Accuracy ของ Gradient)

---

### 🎯 1.2 แผนการเลือก Baselines สำหรับทดลอง (3-Tier Baseline Strategy)

เพื่อพิสูจน์ประสิทธิภาพของ **Middleware ของเรา** (Proactive Memory-Aware บน Commodity Hardware) เราแบ่งการเปรียบเทียบ Baseline ออกเป็น **3 ระดับหลัก** เพื่อโชว์จุดเด่นที่แตกต่างกันในการทดลอง:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                 👑 Our Proposed Thesis Middleware                        │
│   (Local In-Node, Proactive Memory-Aware, Rollback Safety on Commodity)  │
└────────────────────┬────────────────────┬───────────────────────────────┘
                     │                    │
                     ▼                    ▼
     ┌───────────────────────┐    ┌─────────────────────────┐
     │ Tier 1: Standard      │    │ Tier 2: SOTA Direct     │
     │ Framework Baselines   │    │ Auto-Tuning Baselines   │
     ├───────────────────────┤    ├─────────────────────────┤
     │ • PyTorch DataLoader  │    │ • MinatoLoader ('26) ⭐ │
     │ • tf.data (AUTOTUNE)  │    │ • Plumber (LP Model)    │
     │                       │    │ • DLCache (ω rate)      │
     └───────────────────────┘    └─────────────────────────┘
                     │
                     ▼
     ┌──────────────────────────────────────────────────────┐
     │ Tier 3: Structural / Special Format Baselines        │
     ├──────────────────────────────────────────────────────┤
     │ • FFCV / SPDL (แสดงจุดขายไม่ต้องเปลี่ยน Dataset/Python) │
     └──────────────────────────────────────────────────────┘
```

1. **Tier 1: Standard Framework Baselines (ระบบมาตรฐานในอุตสาหกรรม)**
   - **PyTorch DataLoader (Standard):** Baseline พื้นฐานที่สุด (`num_workers` คงที่ 2, 4, 8) ชี้ให้เห็นปัญหา Data Stall หรือ Risk of OOM
   - **`tf.data` (AUTOTUNE):** Dynamic Tuning Baseline ของ Google (Gradient Descent + M/M/1/k Queueing Model) เพื่อแสดงข้อจำกัดเรื่องเสี่ยงเกิด OOM บนเครื่อง RAM จำกัด
2. **Tier 2: SOTA Direct Auto-Tuning Baselines (งานวิจัยเป้าหมายระดับเดียวกัน)**
   - **MinatoLoader (EuroSys '26) ⭐ [SOTA หลัก]:** ใช้ Fast-Slow Queue + P75 Profiling (เน้น HPC RAM 512GB) เปรียบเทียบกับงานของเราที่จัดการ RAM ได้ปลอดภัยกว่าบน **Commodity Hardware (16-32GB RAM)**
   - **Plumber (2022):** Linear Programming Model Baseline
   - **DLCache (2023):** Adaptive Worker Tuning Baseline ปรับ `num_workers` ตาม $\omega = t_{fetch}/t_{req}$ พร้อมประเมิน CPU/RAM ก่อน spawn
3. **Tier 3: Structural / Format Baselines (แสดงความสะดวกในการใช้งาน)**
   - **FFCV / SPDL:** เปรียบเทียบเชิงสถาปัตยกรรมว่า **"งานของเราเป็น Middleware สวมทับ PyTorch เดิมได้ทันที ไม่ต้องแปลง Dataset เป็น `.ffcv` หรือต้องอัปเกรดเป็น Python 3.13 Free-threading"**

_(หมายเหตุ: ระบบอื่นๆ ในตาราง Taxonomy เช่น `ConcurrentDataLoader` ใส่ไว้ในฐานะ Static Async Concurrency Baseline เพื่อความสมบูรณ์ของหมวดหมู่ Taxonomy ไม่ใช่ Baseline หลักในการทดลอง)_

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
                    │  2. Worker Processes = 6 (Async I/O Overlapping)   │
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

- [[01_Research_Idea]] — โครงร่างแผนวิทยานิพนธ์ฉบับปรับปรุง
- [[/01_Knowledge_Base/02_Notes/MinatoLoader_Citation_and_Foundation_Analysis.md|MinatoLoader Analysis]] — การวิเคราะห์เปเปอร์ SOTA MinatoLoader
- [[/01_Knowledge_Base/02_Notes/Data_Pipeline_Bottlenecks_and_PyTorch_Limitations.md|Data Pipeline Bottlenecks]] — สรุปข้อจำกัด PyTorch DataLoader และ SOTA Papers
