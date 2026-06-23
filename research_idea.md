---
tags:
  - thesis
  - blueprint
  - plan
aliases:
  - Thesis_Plan
date: 2026-06-23
---
# 📝 Master Thesis Blueprint: Adaptive Data Loading Framework

## 1. ข้อมูลทั่วไปของวิทยานิพนธ์

- **ชื่อหัวข้อ (Working Title):** Hardware-Aware Adaptive Data Loading Framework for Large-Scale Tensor Data Training
- **ระดับการศึกษา:** ปริญญาโท วิทยาการคอมพิวเตอร์ (Computer Science)
- **โดเมนงานวิจัย:** Systems for Machine Learning / High-Performance Computing (HPC)

## 2. ที่มาและความสำคัญของปัญหา (Problem Statement)

การโหลดข้อมูล (Data Ingestion) เป็นคอขวด (Bottleneck) ที่สำคัญที่สุดในการเทรนโมเดล Deep Learning ขนาดใหญ่ในปัจจุบัน ปัญหานี้แสดงออกใน 2 รูปแบบที่แตกต่างกันอย่างสิ้นเชิงตามสภาพแวดล้อมของฮาร์ดแวร์:

- **Scenario A: เครื่องคอมพิวเตอร์ทั่วไป (Commodity Hardware - RAM Constrained)**
  - **ปัญหา:** เกิด **OS Page Cache Thrashing** เมื่อ Dataset มีขนาดใหญ่กว่า RAM (เช่น ImageNet 100GB+ บนเครื่อง RAM 16GB) ระบบปฏิบัติการต้องเตะข้อมูลเก่าทิ้งและอ่านดิสก์ใหม่ทุก Epoch
  - **ผลกระทบ:** ดิสก์ทำงานหนัก (I/O Bound) ทำให้ `DataLoader_Wait_Sec` พุ่งสูง ส่งผลให้เกิด GPU Starvation (GPU ว่างงานรอข้อมูล)
- **Scenario B: เครื่องเซิร์ฟเวอร์ประสิทธิภาพสูง (HPC - In-Memory Ingestion)**
  - **ปัญหา:** เกิด **Cold-Start Latency** แม้จะมี RAM มหาศาล (เช่น 2TB) เพื่อทำ In-Memory Training แต่การโหลดไฟล์โครงสร้างซับซ้อน (เช่น Spatiotemporal NetCDF ขนาด 1.6TB) เข้า RAM ในครั้งแรก ต้องเจอกับ Overhead ของ Single-threaded parsing และ Decompression
  - **ผลกระทบ:** เสียเวลารอโหลดข้อมูลเป็นชั่วโมงก่อนที่โมเดลจะได้เริ่มเทรน (Ingestion Bottleneck) ขัดขวาง Productivity ของนักวิจัย

## 3. วัตถุประสงค์และจุดขายหลัก (Core Contributions)

เพื่อแก้ปัญหาคอขวดทั้ง 2 รูปแบบ งานวิจัยนี้จะพัฒนา **Python Library Framework** ที่ทำหน้าที่เป็นตัวกลาง (Middleware) สวมทับ PyTorch DataLoader เดิม โดยมีจุดเด่นคือ:

1. **Zero-Touch Automation:** ผู้ใช้ไม่ต้องมีความรู้ระดับ System เพื่อตั้งค่า `num_workers` หรือ `prefetch_factor` ระบบจะวิเคราะห์ฮาร์ดแวร์และจัดการให้เอง
2. **Heuristic-Based Auto-Tuning:** ระบบสามารถตัดสินใจปรับเปลี่ยนกลยุทธ์การโหลดข้อมูลได้เองระหว่างรัน (On-the-fly) ตามกฎ (Rules) ที่กำหนดไว้
3. **Robust Fallback/Rollback Mechanism:** มีระบบความปลอดภัย หาก Framework ปรับจูนค่าแล้วประสิทธิภาพแย่ลง จะทำการถอยกลับ (Rollback) ไปใช้ค่าดั้งเดิมโดยอัตโนมัติ

## 4. ขอบเขตของงานวิจัย (Scope & Limitations)

เพื่อป้องกันไม่ให้โจทย์กว้างเกินไปและสามารถทำสำเร็จได้ตามกำหนด:

- **ประเภทข้อมูลที่รองรับ (In-Scope):** เน้นเฉพาะข้อมูลประเภท **Dense Multi-dimensional Tensors / Arrays** เช่น ไฟล์รูปภาพสเกลใหญ่ (ImageNet) และข้อมูลวิทยาศาสตร์/พยากรณ์อากาศแบบ Spatiotemporal (`.nc`, `.h5`)
- **ประเภทข้อมูลที่ไม่รองรับ (Out-of-Scope):** ข้อมูลที่มีความยาวไม่คงที่ (Variable-length sequences) เช่น Text/NLP และข้อมูลกราฟ (Graph Data)

## 5. สถาปัตยกรรมของ Framework (System Architecture)

การออกแบบจะใช้หลักการแบ่งแยกหน้าที่ความรับผิดชอบของโดเมนระบบอย่างชัดเจน โดยแบ่งการทำงานออกเป็น 3 ขั้นตอนหลัก:

### Phase 1: Static Hardware Profiler & User Hints

- เมื่อถูกเรียกใช้งาน Framework จะตรวจสอบสภาพแวดล้อมทันที (CPU Cores, Available RAM, สถาปัตยกรรม GPU)
- รับคำใบ้ (Data Hint) จากผู้ใช้ผ่าน API แบบ Data validation (เพื่อให้การระบุประเภทข้อมูล เช่น `PATTERN_RANDOM_EACH` หรือ `PATTERN_CONTIGUOUS_CHUNKS` เข้มงวดและไม่มีข้อผิดพลาดแต่ต้น)
- อ่านค่าเหล่านี้และเก็บเป็น Read-only configuration (เช่น การใช้ Immutable Data Structures) เพื่อสร้าง **Baseline State** ไว้เป็นจุดอ้างอิงในการ Rollback หากจำเป็น

### Phase 2: Policy Selector & Initialization

- ระบบวิเคราะห์เงื่อนไขตั้งต้น (เช่น $Dataset > RAM$ หรือไม่) เพื่อเลือกกลยุทธ์แรกเริ่ม
- สร้าง Custom Sampler ส่งให้ PyTorch DataLoader แทนที่จะปล่อยให้ PyTorch ทำงานแบบ Default

### Phase 3: Dynamic Runtime Auto-Tuner

- มี Background Monitor ทำงานคู่ขนาน คอยจับ Metrics เช่น `GPU_Idle_Starvation_Sec`, `Total_GPU_Compute_Sec`, และ `Peak_RAM_GB`
- **Minor Tuning:** หากสถาปัตยกรรมภายในอนุญาต จะอัปเดตตัวแปรแชร์ความจำที่ Custom Sampler เรียกใช้ (เช่น ปรับ `chunk_size`) เพื่อให้มีผลใน Batch ถัดไปโดยไม่ต้องรีสตาร์ท Process
- **Major Tuning:** หากตรวจพบวิกฤต เช่น GPU Starvation > 20% ระบบจะรอให้จบ Epoch แล้วจึงสร้าง DataLoader ตัวใหม่ด้วยพารามิเตอร์ที่เหมาะสมกว่า
- **Rollback Logic:** วัด Throughput ของ Epoch ปัจจุบันเทียบกับ Epoch ก่อนหน้า หากค่าต่ำลงเกินเกณฑ์ที่กำหนด จะสั่งรันคำสั่งย้อนกลับไปใช้ Baseline Configuration ทันที

## 6. คู่เทียบและตัวชี้วัด (Baselines & Evaluation Metrics)

เพื่อพิสูจน์คุณค่าของงานวิจัยในเล่มวิทยานิพนธ์ จะต้องทำการเปรียบเทียบผลลัพธ์ (Empirical Benchmarking):

- **คู่เทียบ (Baselines):**
  1. Default PyTorch DataLoader (การตั้งค่าแบบ Manual ทั่วไป)
  2. State-of-the-art Frameworks ที่มีอยู่ (เช่น MinatoLoader หรือกลยุทธ์พื้นฐานของ TensorStore)
- **ตัวชี้วัดความสำเร็จ (Metrics):**
  - **Throughput:** จำนวนภาพ/ตัวอย่าง ที่ประมวลผลได้ต่อวินาที (Images/Sec)
  - **Resource Utilization:** อัตราส่วนที่ GPU ทำงานจริง (`GPU_Busy_Ratio_All`) โดยตั้งเป้าให้เข้าใกล้ 95-99%
  - **Cold-Start Latency:** เวลาที่ใช้ตั้งแต่กดรันคำสั่ง จนกระทั่ง GPU เริ่มคำนวณ Batch แรก (โดยเฉพาะในเคส NetCDF ขนาด 1.6TB)

## 7. แผนการดำเนินงานและการออกแบบโค้ด (Software Design Principles)

- **Drop-in Replacement:** ออกแบบ API ของ Library ให้สวมเข้ากับโค้ดเดิมของผู้ใช้ได้ด้วยการเปลี่ยนโค้ดเพียง 1-2 บรรทัด
- **Modularity:** เขียนโค้ดในลักษณะที่ระบบ Heuristic Rules แยกออกจากตัว Core DataLoader เพื่อให้ในอนาคต (Future Work) นักวิจัยคนอื่นสามารถมาเขียน Rule เสียบเพิ่มได้โดยไม่ต้องรื้อระบบ
- **Safe State Management:** กฎเหล็กคือห้ามทำลายสถานะของคิวข้อมูล (Sampler state) และหลีกเลี่ยงการ Spawning Process ใหม่กลาง Epoch เพื่อป้องกัน Overhead ทำลายประสิทธิภาพระบบ
