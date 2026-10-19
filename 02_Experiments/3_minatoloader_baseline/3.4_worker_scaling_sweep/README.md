# 3.4 Worker Scaling Sweep Benchmark (Workers 1 to 8)

การทดลองสเกลจำนวน Worker อย่างละเอียดตั้งแต่ **1 ถึง 8 Workers** เพื่อค้นหาจุดเปลี่ยนผัน (**Inflection Point**) และพิสูจน์สาเหตุที่ 8 Workers ทำงานช้ากว่า 4 Workers ภายใต้สภาวะ **Heavy 3D Medical Preprocessing**

---

## 🔬 สมมติฐานและบริบททางฮาร์ดแวร์ (Hardware Hypothesis)
- **เครื่องทดลอง:** ซีพียู **Intel Core i5-12450HX** (Alder Lake Architecture)
  - 🚀 **4 Performance-Cores (P-Cores):** คอร์กำลังสูง รองรับ Hyper-threading
  - 🐢 **4 Efficient-Cores (E-Cores):** คอร์ประหยัดพลังงาน ความเร็วต่ำ
- **สมมติฐานทางวิทยาศาสตร์:**
  1. การเพิ่ม Worker จาก **1 ➔ 4** จะช่วยเพิ่ม Throughput อย่างชัดเจน เพราะระบบสามารถกระจายงาน Heavy Augmentation (`ElasticDeformation`, `HeavyGaussianBlur`) ลงบน **4 P-Cores** ได้อย่างเต็มประสิทธิภาพ
  2. เมื่อเพิ่ม Worker เกิน 4 (คือ **5 ถึง 8 Workers**): งานจะถูกผลักไปรันบน **E-Cores** ซึ่งคำนวณ 3D Spline ได้ช้ากว่ามาก กลายเป็นตัวถ่วงแถว (Straggler / Head-of-Line Blocking) ประกอบกับ Context Switching แย่งแคช L2/L3 ทำให้ Throughput รวมเริ่มตกต่ำลง

---

# 3.4 Dual-Perspective Worker Scaling Sweep Benchmark (Workers 1 to 8)

การทดลองสเกลจำนวน Worker อย่างละเอียดตั้งแต่ **1 ถึง 8 Workers** เพื่อค้นหาจุดเปลี่ยนผัน (**Inflection Point**) และพิสูจน์สาเหตุที่ 8 Workers ทำงานช้ากว่า 4 Workers ภายใต้สภาวะ **Heavy 3D Medical Preprocessing**

---

## 🔬 สองมุมมองการทดลอง (Two Experimental Perspectives)

เพื่อให้การวิเคราะห์ในวิทยานิพนธ์มีความสมบูรณ์เชิงวิชาการ เราแบ่งการทดสอบออกเป็น 2 โหมด:

### 1. Mode B: Continuous Streaming Benchmark (`run_worker_sweep.py`)
- **ลักษณะการทำงาน:** เทรนต่อเนื่อง 3 Epochs โดยไม่มีการ Pause เพื่อทำ Validation
- **สิ่งที่วัด:** Steady-state DataLoader Throughput แท้จริง และขีดจำกัดการคำนวณสดของ CPU
- **ผลลัพธ์:**
  - Minato สเกลจาก **4.08 img/s (1W)** $\rightarrow$ **15.06 img/s (4W)** $\rightarrow$ **19.77 img/s (8W)**
  - อธิบายผลของสถาปัตยกรรม CPU Hybrid (4 P-Cores + 4 E-Cores) ที่เริ่มอิ่มตัว (diminishing returns) เมื่อเกิน 4 Workers

### 2. Mode A: With Validation Pause Benchmark (`run_worker_sweep_with_val.py`)
- **ลักษณะการทำงาน:** รัน 2 Epochs โดยมี Full Validation (41 CT scans) คั่นระหว่าง Epoch
- **สิ่งที่วัด:** พฤติกรรมของ Multi-process Queue เมื่อมีช่วงเวลาว่างให้แอบดึงข้อมูลล่วงหน้า (Pre-buffering) และผลกระทบของ IPC Lock Contention
- **ผลลัพธ์:**
  - ที่ **W=4**: ผ่านการประมวลผลล่วงหน้าอย่างราบรื่น ได้ Throughput พุ่งแตะ **~28.8 img/s** (Wait latency 7.5 ms)
  - ที่ **W=8**: เกิด IPC Queue Lock Contention อย่างรุนแรง ทำให้ Throughput ตกลงเหลือ **~20.6 img/s** (Wait latency พุ่งเป็น 28.8 ms)

---

## 📊 แดชบอร์ดสรุปผลคู่ขนาน (Dual-Perspective Dashboard)
- รันสคริปต์เพื่อสร้างกราฟระดับสิ่งพิมพ์:
```bash
python3 plot_dual_perspective.py
```
- กราฟจะถูกบันทึกไว้ที่: `results/plots/dual_perspective_comparison.png`

---

## 🚀 วิธีการรัน

### รัน Continuous Streaming (Mode B):
```bash
python3 run_worker_sweep.py --workers 1,2,3,4,5,6,7,8 --epochs 3
```

### รัน With Validation (Mode A):
```bash
python3 run_worker_sweep_with_val.py --workers 6,5,7,2,3,1 --epochs 2
```

