# 3.3 Dynamic WorkerScheduler & Adaptive Timeout (Our Custom Implementation)

การทดลองส่วนที่สามที่เป็น **งานพัฒนาฟีเจอร์ที่ขาดหายไปของเปเปอร์ MinatoLoader ด้วยตนเอง (Our Contribution)** ได้แก่ ระบบปรับจำนวนคนงานอัตโนมัติ (**Worker Scheduler**) และระบบตั้งค่าเพดานเวลาไดนามิก (**Adaptive Timeout P75**)

---

## 🎯 วัตถุประสงค์ (Objective)
1. เติมเต็มช่องโหว่ของเปเปอร์ MinatoLoader (EuroSys 2026) ที่ใน GitHub ขาดโค้ด Dynamic Auto-tuning
2. พัฒนาระบบ `WorkerScheduler`:
   - ใช้ `psutil` ตรวจจับ CPU Utilization แบบ Real-time
   - ปรับจำนวน Process Workers ขึ้น-ลง (Dynamic Range: 2 – 8 workers) ตามโหลด
3. พัฒนาระบบ `Adaptive Timeout`:
   - คำนวณค่า Percentile ที่ 75 (P75) ของเวลาเตรียมรูปภาพทุกๆ 5 Batch
   - นำค่า P75 มาตั้งเป็นเกณฑ์แยกรูปช้าลง Slow Queue อัตโนมัติ (แทนการ Hardcode 2.0s)

---

## ⚙️ การตั้งค่า 2 คอนฟิกูเรชัน (Configurations)
| Config | ระบบ (System) | Worker Scheduler | Adaptive Timeout | Max Workers |
|:---|:---|:---:|:---:|:---:|
| **C1** | Minato + Worker Scheduler | ✅ Dynamic (2 ➔ 8) | ❌ Fixed (2.0s) | 8 |
| **C2** | Minato + Full Auto-tune | ✅ Dynamic (2 ➔ 8) | ✅ Adaptive P75 | 8 |

---

## 📊 ผลการทดลองและ Insights สำคัญของวิทยานิพนธ์ (Key Insights)
- ✅ **ระบบทำงานได้จริงตามทฤษฎี:** กราฟ `worker_scaling_timeline.png` แสดงให้เห็นชัดเจนว่าจำนวน Worker ไต่ระดับขึ้น-ลงตามโหลด CPU ได้อย่างแม่นยำ
- ⚠️ **ประสิทธิภาพรวมลดลงเมื่อเทียบกับ B1 (Fixed 4W):**
  - **Overhead ของ Python Multiprocessing:** การ Fork / Spawn และ Kill Process กลางคันสร้าง Overhead มหาศาล
  - **Sawtooth Effect (Epoch Reset):** ทุกครั้งที่เริ่ม Epoch ใหม่ Worker จะถูกรีเซ็ตกลับไปที่ 2 ทำให้ต้องเสียเวลาไต่ระดับใหม่
  - **IPC & GIL Locks:** การส่งสถิติ P75 และ Queue Monitoring ข้าม Process ผ่าน IPC ทำให้ติด GIL
- 💡 **บทสรุปสำหรับวิทยานิพนธ์ (Core Thesis Contribution):**
  การทำ Dynamic Auto-tuning เพื่อป้องกัน CPU Overload มีประโยชน์ในเชิงทฤษฎี **แต่ไม่ควรทำในระดับ Python Layer** โดยเสนอแนะว่าควรนำลอจิกนี้ลงไปฝังในระดับ **C++ Core Engine ของ PyTorch** หรือรอ Free-Threading ใน Python 3.13+

---

## 📁 โครงสร้างโฟลเดอร์
```
3.3_adaptive_scheduler/
├── worker_scheduler.py          # คลาส ScheduledAsynchronousLoader พร้อมตรรกะ Auto-tuning
├── run_scheduled_experiments.py # สคริปต์รัน C1 และ C2
├── run_adaptive_scheduler.sh    # สคริปต์รันรวดเดียว C1 + C2 พร้อมวาดกราฟ
├── plot_worker_scaling.py       # วาดกราฟไทม์ไลน์การปรับจำนวนคนงาน (Worker Scaling)
├── plot_timelines.py            # วาดกราฟ GPU Timeline ราย Batch ของ C1 และ C2
├── evaluation_cases.txt         # ไฟล์รายชื่อ Case สำหรับแบ่ง Train/Val
└── results/
    ├── heavy_out_C1/            # ผลลัพธ์ C1: Minato + WorkerScheduler
    ├── heavy_out_C2/            # ผลลัพธ์ C2: Minato + Scheduled + Adaptive
    └── plots/                   # กราฟสรุปผล (Worker scaling, Timelines)
```
