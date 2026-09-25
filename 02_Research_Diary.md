---
tags:
  - diary
  - daily-log
---

# 📓 Research Diary & Daily Log

ไฟล์นี้ (research_diary.md) เอาไว้ใช้ **"จดบันทึกแบบไม่เป็นทางการ (Logbook)"** ครับ
เวลาทำธีสิส เรามักจะลืมว่า "อาทิตย์ที่แล้วเราแก้บั๊กอะไรไปนะ?" หรือ "ทำไมวันนั้นถึงตัดสินใจเปลี่ยนจาก Mobilenet เป็น ResNet18?" ไฟล์นี้คือที่สำหรับจดสิ่งเหล่านั้นครับ

## 📝 วิธีใช้งานแบบ Obsidian

แนะนำให้จดโดยใส่วันที่นำหน้าแบบนี้ (สร้างเป็น header ย่อย หรือ bullet point ไปเรื่อยๆ จากบนลงล่าง):

### 🗓️ 2026-06-23

- **สิ่งที่ทำ:** รันและวิเคราะห์ผลการทดลอง Experiment Scaling เพื่อหา Data Bottleneck
- **สิ่งที่ค้นพบ:**
  - เกิด GPU Starvation หนักมากถ้า Workers ต่ำ
  - ถ้า Data เล็กจะถูกดึงเข้า OS Page Cache (Dry run)
  - ถ้าใช้โมเดลช้าอย่าง ResNet18 คอขวดจะย้ายไปที่ Compute ต่อให้ใช้ W=2 ก็ตาม
- **อ้างอิงผลลัพธ์:** เข้าไปอ่านสรุปแบบเต็มได้ที่ [[experiment_summary|Experiment_Scaling_Analysis]]
- **Next Step (แผนพรุ่งนี้):** อาจจะเริ่มลองแก้ Custom Sampler เพื่อปรับพารามิเตอร์แบบ On-the-fly ตามที่เขียนไว้ใน [[/01_Research_Idea.md|Research_Idea]]

---

### 🗓️ 2026-07-09

- **สิ่งที่ทำ:** อ่านเปเปอร์ 4 ฉบับ (tf.data, Plumber, CoorDL, Pollux) และหารือกับ LLM เรื่องข้อจำกัดเชิงสถาปัตยกรรมของ PyTorch DataLoader (GIL, IPC Overhead)
- **สิ่งที่ค้นพบ:**
  - การจูน `num_workers` ของ PyTorch กลางอากาศทำได้ยากเพราะ Overhead ของ Python Multiprocessing
  - วงการวิจัยส่วนใหญ่ไปใช้ C++ Drop-in replacement (FFCV, DALI) ซึ่งทำให้มี Learning Curve
  - เป็นช่องโหว่งานวิจัยที่ยอดเยี่ยมในการทำ Python-native Middleware ที่ปรับจูนข้าม Epoch หรือปรับระดับ Chunk ภายใน Shared Memory ได้โดยไม่ต้องรีสตาร์ท Process กลาง Epoch
- **อ้างอิงผลลัพธ์:** สร้างบันทึกความรู้ใหม่ไว้ที่ [[/01_Knowledge_Base/02_Notes/Data_Pipeline_Bottlenecks_and_PyTorch_Limitations.md|Data_Pipeline_Bottlenecks_and_PyTorch_Limitations]] และเพิ่มจุดเด่นเรื่อง Research Gap ใน [[/01_Research_Idea.md|Research_Idea]]
- **Next Step:** เริ่มออกแบบ Architecture ของ Middleware ตัวนี้ โฟกัสที่ Heuristic Rules หรือ Data Hint API เบื้องต้น

---

### 🗓️ 2026-08-05

- **สิ่งที่ทำ:** ประชุมอัปเดตความคืบหน้ากับอาจารย์ที่ปรึกษา และนำเปเปอร์ใหม่ที่ได้รับคำแนะนำ (MinatoLoader, DLCache, ConcurrentDataloader, Synergy, Plumber) มาวิเคราะห์เชิงลึกเพื่อหาแนวทางออกแบบ Dynamic Auto-Tuner
- **สิ่งที่ค้นพบ (Insights):**
  - **การปรับจูนแบบไดนามิกเป็นไปได้และจำเป็น:** เปเปอร์เหล่านี้ยืนยันว่าการประเมินเพื่อหาจำนวน Worker แบบ Real-time สามารถทำได้อย่างแม่นยำด้วยคณิตศาสตร์ (เช่น ทฤษฎีคิวของ Synergy, สมการ $\omega$ ของ DLCache, และ Linear Programming ของ Plumber) โดยพิจารณาจาก "อัตราการดึงข้อมูล" (Arrival Rate) เทียบกับ "ความเร็วที่ GPU เทรน" (Service Rate)
  - **แสงสว่างสำหรับ Python-native:** ConcurrentDataloader ยืนยันว่าแม้ Python จะมีข้อจำกัดเรื่อง GIL แต่สำหรับงานโหลดข้อมูลที่เป็น I/O-bound การสลับไปใช้ Concurrency (Asyncio) หรือ Thread Pool ก็สามารถทะลวงคอขวดได้โดยไม่ต้องเปลี่ยนไปเขียน C++ แบบ Framework อื่นๆ
  - **ตัวแปรชี้วัด (Metrics) ที่ฉลาดขึ้น:** เราไม่ควรตั้งค่า Timeout แบบสุ่ม แต่ควรทำ Profiling (P75 ของ MinatoLoader) และการสลับ/แคชข้อมูลต้องคิด "ต้นทุนเวลาในการเตรียมข้อมูล" ร่วมด้วยเสมอ
- **อ้างอิงผลลัพธ์:** สรุปสาระสำคัญของแต่ละเปเปอร์พร้อมแนบไฟล์ PDF ต้นฉบับไว้ที่ [[/01_Knowledge_Base/02_Notes/Data_Pipeline_Bottlenecks_and_PyTorch_Limitations.md|Data_Pipeline_Bottlenecks_and_PyTorch_Limitations]]
- **ปัญหาที่เจอ:** ปัจจุบันเรามีหลายทฤษฎีในมือ (Hill-Climbing, Linear Programming, Queuing Theory, สมการ $\omega$) ความท้าทายต่อไปคือการ "คัดเลือก" หรือ "ผสาน" สมการเหล่านี้ให้เหมาะสมกับเงื่อนไขฮาร์ดแวร์ของเรามากที่สุด (RAM จำกัด vs HPC)
- **Next Step:** นำสมการ/อัลกอริทึมเหล่านี้มากางเทียบกันเพื่อเลือก **Core Algorithm** และเริ่มร่างโครงสร้าง Architecture ว่าจะเสียบ Middleware ตัวนี้เข้ากับ `DataLoader` ของ PyTorch โดยไม่ให้กระทบโค้ดเดิมของผู้ใช้ได้อย่างไร

---

### 🗓️ 2026-09-12

- **สิ่งที่ทำ:** ประชุมอัปเดตและรับโจทย์ใหม่จากอาจารย์ที่ปรึกษา ( Taxonomy การทำ Auto-tune, เลือก Scope Strategy, วิเคราะห์ Case Study Weather Time-Series)
- **สิ่งที่ค้นพบ/ข้อสรุป:**
  - **Taxonomy:** จัดกลุ่มการ Auto-tune ออกเป็น 4 มิติ (Tuning Target, Control Mechanism, Hardware Environment, Data Access Pattern) และระบุตำแหน่งงานวิจัยของเราเป็น _Proactive Memory-Aware Auto-Tuning Middleware on Commodity Hardware_
  - **Scope Strategy:** เลือกแนวทาง **Hybrid / Pattern-Aware Adaptive Middleware** ที่ปรับพฤติกรรมอัตโนมัติตามขนาด Dataset และ RAM Budget (In-Memory Cache vs Sliding Prefetch Buffer Stream Read vs CPU Worker Scaling)
  - **Case Study Analysis (Weather Time-Series):** แก้ปัญหาความขัดแย้งระหว่างการอัด Dataset ทั้งหมดเข้า RAM (Cold Start นาน + เสี่ยง OOM) กับการอ่านทีละไฟล์ (I/O Bottleneck) ด้วยการใช้ **Dynamic Prefetch Buffer Window + Async Multi-Worker Reading** ในระดับ Middleware
- **อ้างอิงผลลัพธ์:** สร้างเอกสารวิเคราะห์และสรุปแบบละเอียดไว้ที่ [[/01_Knowledge_Base/02_Notes/AutoTuning_Taxonomy_and_Advisor_Feedback.md|AutoTuning_Taxonomy_and_Advisor_Feedback]]
- **Next Step:** นำแผนภาพ Taxonomy และ Hybrid Pattern-Aware Strategy ไปนำเสนออาจารย์ในรอบถัดไป และเริ่มร่าง Proof of Concept (PoC) สำหรับ Pattern Detection Module

---

### 🗓️ 2026-09-25

- **สิ่งที่ทำ:** จัดระเบียบลำดับขั้น (Lineage Tree & Layer Breakdown) ของเปเปอร์ทั้งหมด 25+ ระบบ จำแนกต้นกำเนิดแนวคิด (Materialized Views, Adaptive Query Processing, Event-Driven Architecture) และอัปเดตสถานะไฟล์ PDF ทั้งหมดใน Vault
- **สิ่งที่ค้นพบ/ข้อสรุป:**
  - **PDF Tracking:** ดาวน์โหลดและเก็บไฟล์ PDF ในเครื่องครบสมบูรณ์ทุกรายการ (รวม DL Data Loaders ล่าสุด Cedar, FFCV, SPDL, Joader และเปเปอร์ Survey ทั้ง Lotus และ Empirical Study on Low GPU Utilization)
  - **Sub-reference Guidance:** เปเปอร์ย่อยระดับฐานข้อมูลดั้งเดิม (เช่น Agrawal, Heydon, Hellerstein) ไม่จำเป็นต้องดาวน์โหลด PDF เก็บเพิ่ม ให้ใช้วิธีอ้างอิงผ่านเปเปอร์ Layer 1 หลัก เช่น Pecan (ATC '24) และ Cachew (ATC '22)
  - **Survey & Empirical Study Papers:** ผูกลิงก์ไฟล์ PDF ของ `lotus.pdf` และ `Empirical Study on Low GPU Utilization.pdf` ไว้เรียบร้อย สำหรับใช้อ้างอิง **Problem Motivation ในบทที่ 1 และ 2**
- **อ้างอิงผลลัพธ์:** อัปเดตตารางสรุปเปเปอร์ใน [[/01_Knowledge_Base/02_Notes/AutoTuning_Taxonomy_and_Advisor_Feedback.md|AutoTuning_Taxonomy_and_Advisor_Feedback]] และคู่มือดาวน์โหลดใน [[/01_Knowledge_Base/01_Papers/README_PDF_Download_Checklist.md|README_PDF_Download_Checklist]]

