---
tags:
  - knowledge
  - literature-review
  - data-pipeline
  - auto-tuning
aliases:
  - Data_Pipeline_Bottlenecks
date: 2026-07-09
---

# 📚 Literature Review: Data Pipeline Bottlenecks & PyTorch Limitations

จากการทบทวนวรรณกรรม 4 ฉบับล่าสุดและการวิเคราะห์เชิงลึกเกี่ยวกับสถาปัตยกรรมของ PyTorch ได้ข้อสรุปที่สำคัญต่อ [[/01_Research_Idea|โครงร่างวิทยานิพนธ์]] ดังนี้

## 1. สรุปงานวิจัยที่เกี่ยวข้อง (Literature Review)

เอกสารทั้ง 4 ฉบับมุ่งเน้นแก้ปัญหา Data Stalls และคอขวดใน Machine Learning Data Pipelines:

1. **tf.data (Google):** [[tf_data_A_Machin_Learning_Data_Processing_Framework.pdf|อ้างอิงไฟล์ PDF]]
   - **ปัญหา:** GPU ว่างงานรอข้อมูล
   - **ทางแก้:** Framework ที่มีระบบ `AUTOTUNE` ปรับจูนทรัพยากร (CPU, RAM) อัตโนมัติ (Dynamic Auto-tuning) โดยใช้อัลกอริทึมเช่น Hill-Climbing
   - **เบื้องหลัง:** ใช้ C++ และออกแบบเป็นกราฟ ทำให้คำนวณและปรับลด Thread ของแต่ละโหนดได้กลางอากาศโดยไม่มี Overhead
2. **Plumber:** [[PLUMBER_DIAGNOSING_AND_REMOVING_PERFORMANCE_BOTTLENECKS_IN_MACHINE_LEARNING_DATA_PIPELINES.pdf|อ้างอิงไฟล์ PDF]]
   - **ปัญหา:** 62% ของงานใน Google ยังเจอคอขวดจากซอฟต์แวร์ แม้จะใช้ tf.data
   - **ทางแก้:** เครื่องมือวิเคราะห์ที่ครอบทับ tf.data ใช้ Linear Programming วิเคราะห์หาคอขวด (CPU, Disk, Memory) และปรับจูนความขนาน/Prefetch ให้อัตโนมัติ
3. **DS-Analyzer & CoorDL:** [[Analyzing_and_Mitigating_Data_Stalls_in_DNN_Training.pdf|อ้างอิงไฟล์ PDF]]
   - **ปัญหา:** การใช้ OS Page Cache ทำให้เกิด Data Stalls จากการเตะข้อมูลทิ้ง (Thrashing) และอ่านข้อมูลซ้ำซ้อน
   - **ทางแก้:** เสนอไลบรารีแบบ Drop-in replacement (เช่น ร่วมกับ NVIDIA DALI) ที่มี MinIO Cache และประสานงานการดึงข้อมูลระหว่างโหนด
4. **Pollux:** [[Pollux_Co-adaptive_Cluster_Scheduling_for_Goodput-Optimized_Deep_Learning.pdf|อ้างอิงไฟล์ PDF]]
   - **ปัญหา:** ระบบจัดสรรทรัพยากรคลัสเตอร์ไม่ได้สนใจพารามิเตอร์ของการเทรน
   - **ทางแก้:** ระบบ Co-adaptive ที่ปรับจูนทั้ง "ทรัพยากร (GPU)" และ "พารามิเตอร์ (Batch size, Learning rate)" ไปพร้อมกัน โดยใช้มาตรวัด "Goodput" (System Throughput × Statistical Efficiency)

## 2. ข้อจำกัดของ PyTorch DataLoader (The Root Cause)

ทำไมวงการวิจัยถึงยังทำ Auto-tune ให้ PyTorch ไม่ได้เหมือน tf.data?

- **Python GIL (Global Interpreter Lock) คืออะไรและทำไมจึงเป็นปัญหา?:** GIL คือกลไกของ Python ที่ป้องกันไม่ให้หลาย Thread ทำงานพร้อมกันเพื่อความปลอดภัยของ Memory เมื่อการโหลดและประมวลผลข้อมูลต้องการใช้ CPU หนัก (CPU-bound) การใช้ Multithreading จึงไม่ช่วยให้เร็วขึ้น PyTorch DataLoader (ซึ่งเขียนด้วย Python) จึงต้องเลี่ยงปัญหา GIL ด้วยการสร้าง Process ใหม่ทั้งหมด (Multiprocessing ผ่าน `num_workers > 0`) แทน
- **IPC Overhead:** การส่ง Tensor ข้าม Process และแปลงข้อมูล (Serialization/Pickling) สร้าง Overhead มหาศาล
- **ไม่สามารถจูนกลางอากาศได้:** การสั่งยุบหรือสร้าง Process ใหม่กลาง Epoch ทำให้ Memory Queue พัง และเกิด Overhead จนระบบค้าง

## 3. ช่องว่างงานวิจัย (Research Gap) สำหรับวิทยานิพนธ์

ทำไมจึงควรลงทุนอุดช่องโหว่ให้ PyTorch แทนที่จะย้ายไปใช้ TensorFlow (tf.data)?

- **Market Dominance:** ปัจจุบัน PyTorch ครองส่วนแบ่งในวงการวิจัยระดับท็อป และโมเดลบน Hugging Face กว่า **85%** ([อ้างอิง: PyTorch vs TensorFlow 2025](https://leapcell.io/blog/tensorflow-vs-pytorch-a-comparative-analysis-for-2025)) หากสามารถสร้าง Framework ที่แก้ปัญหานี้ได้ จะสร้าง Impact ให้ผู้ใช้งานมหาศาล
- **ความพยายามที่ล้มเหลว (TorchData/DataLoader2):** ทีมงาน PyTorch ตระหนักถึงปัญหานี้และเคยสร้างโปรเจกต์ `TorchData` เพื่อพยายามแก้ปัญหา Data Pipeline แต่วิธีการกลับเพิ่มความซับซ้อนเกินไป จนท้ายที่สุดโปรเจกต์ถูกยุติบทบาทการพัฒนาลง โดยทีมงานได้ประกาศใน [GitHub Issue #1196](https://github.com/meta-pytorch/data/issues/1196) ว่า:
  > _"As of July 2023, we have paused active development on TorchData and have paused new releases. We have learnt a lot from building it and hearing from users, but also believe we need to re-evaluate the technical design and approach given how much the industry has changed since we began the project. During the rest of 2023 we will be re-evaluating our plans in this space."_

ด้วยเหตุนี้ งานวิจัยส่วนใหญ่จึงหลีกเลี่ยงปัญหานี้โดยการ "เขียนระบบใหม่ด้วย C++" แทน (เช่น FFCV, NVIDIA DALI) ซึ่งสร้างภาระการเรียนรู้ (Steep Learning Curve) และผู้ใช้ต้องแปลงไฟล์ Dataset
นี่คือโอกาสทองในการสร้าง **Python-native Middleware** ที่เพิ่มความสามารถ Auto-tuning ให้ PyTorch DataLoader ดั้งเดิม โดยไม่ต้องแปลงไฟล์:

- **On-the-fly Tuning (ระหว่าง Batch):** หลีกเลี่ยงการเปลี่ยน `num_workers` กลาง Epoch แต่ให้สถาปัตยกรรมปรับจูนตัวแปรแชร์ความจำในระดับ Python เช่น `chunk_size` หรือ `batch_size` ภายใน Shared Memory แทน
- **Epoch-boundary Tuning (ข้าม Epoch):** หากระบบคำนวณแล้วว่าต้องเปลี่ยน `num_workers` หรือ `prefetch_factor` จะทำการรอให้จบ Epoch เพื่อเคลียร์ Memory อย่างสะอาดหมดจด แล้วสร้าง DataLoader ตัวใหม่ เพื่อเลี่ยง Overhead กลาง Epoch

## 4. ข้อพิจารณาและ Trade-off ของแต่ละแนวทาง

การแก้ปัญหา Data Pipeline มีข้อแลกเปลี่ยน (Trade-off) ที่ต้องชั่งน้ำหนักเสมอ:

- **C++ Drop-in (FFCV, DALI) vs Python Middleware (งานวิจัยนี้):**
  - _C++ Drop-in:_ ได้ความเร็วสูงสุด ขจัด Overhead สิ้นเชิง แต่แลกมาด้วยความยืดหยุ่น (Flexibility) ที่ลดลง ผู้ใช้เขียน Custom Augmentation ยาก และต้องแปลงฟอร์แมตข้อมูล (เช่น `.ffcv`)
  - _Python Middleware:_ ใช้งานง่าย สวมทับกับ DataLoader เดิมได้ทันที ไม่ต้องแปลงข้อมูล แต่ยังต้องยอมรับข้อจำกัดด้านความเร็วบางส่วนจาก Python IPC Overhead
- **On-the-fly vs Epoch-boundary Tuning:**
  - _On-the-fly (ปรับทันที):_ ปรับตามสถานการณ์ได้ไว (Reactive) แต่ใน Python ทำได้เฉพาะการปรับตัวแปรที่ไม่เปลี่ยนสถานะของ Process (เช่น Chunk size)
  - _Epoch-boundary (รอจบ Epoch):_ สามารถเปลี่ยน `num_workers` และเคลียร์ Memory รั่วไหลได้ปลอดภัย 100% แต่ต้องใช้เวลารอนานกว่าระบบจะเริ่มปรับค่าให้ (Delayed Reaction)

**อ้างอิงและจุดเชื่อมโยง (Internal & External):**

- แนวคิดนี้นำไปเสริมจุดแข็งในหน้าหลัก [[01_Research_Idea]]
- บันทึกการดำเนินการประจำวันใน [[02_Research_Diary]]
- **ข้อมูลเครื่องมือและงานวิจัยที่เกี่ยวข้องอื่นๆ:**
  - [FFCV (Fast Forward Computer Vision)](https://ffcv.io/) - โปรเจกต์แก้ปัญหาคอขวดด้วยการเปลี่ยนไปใช้ C++ และแปลงไฟล์ Dataset
  - [NVIDIA DALI (Data Loading Library)](https://github.com/NVIDIA/DALI) - ไลบรารีสำหรับย้าย Data Preprocessing ไปประมวลผลบน GPU
  - [Joader: A Data-Centric Data Fetching System for Deep Learning](https://scholar.google.com/scholar?q=Joader:+A+Data-Centric+Data+Fetching+System+for+Deep+Learning) - งานวิจัยแก้ปัญหาเมื่อมีการเทรนหลายโมเดลแล้วเกิดการโหลดข้อมูลซ้ำซ้อน
