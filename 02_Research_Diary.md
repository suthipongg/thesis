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

### 🗓️ [YYYY-MM-DD]

- **สิ่งที่ทำ:** ...
- **ปัญหาที่เจอ:** ...
- **Next Step:** ...
