# 🚀 MinatoLoader Baseline & Benchmark Series

ชุดการทดลองและเปรียบเทียบประสิทธิภาพระหว่าง **PyTorch DataLoader** กับ **MinatoLoader (EuroSys 2026)** บนโมเดล 3D-UNet และชุดข้อมูลการแพทย์ KiTS19 โดยแบ่งการทดลองออกเป็น 3 ส่วนหลักอย่างเป็นระเบียบ:

---

## 📑 สารบัญการทดลอง (Experiment Overview)

### 🔹 [3.1 Initial Lightweight Baseline](./3.1_light_baseline)
- **สถานะ:** การทดสอบรอบแรก (Baseline ดั้งเดิมแบบไม่มี Preprocessing หนักๆ)
- **เงื่อนไข:** 2 Workers, Light Augmentation (Flip, Brightness), Batch Size = 2
- **ข้อค้นพบ:** MinatoLoader ทำความเร็วได้พอๆ กับ PyTorch เนื่องจาก Augmentation เบาเกินไป ทำให้ CPU ไม่เกิดคอขวดและ Queue กลไกพิเศษของ MinatoLoader ไม่ได้ถูกเรียกใช้งานจริง

---

### 🔹 [3.2 Heavy Augmentation Comparison](./3.2_heavy_comparison)
- **สถานะ:** การทดลองเปรียบเทียบเชิงลึกเพื่อจำลอง CPU Bottleneck ที่แท้จริง (Fair Benchmark)
- **เงื่อนไข:** Heavy 3D Augmentation (`ElasticDeformation`, `HeavyGaussianBlur`, `GaussianNoise`), Batch Size = 2
- **4 คอนฟิกูเรชัน:**
  - `A1`: PyTorch (4 Workers)
  - `A2`: PyTorch (8 Workers)
  - `B1`: MinatoLoader (4 Workers) — 🏆 **Sweet Spot เร็วที่สุด (~30 img/s)**
  - `B2`: MinatoLoader (8 Workers) — เกิด Context Switching คอขวด (~21 img/s)

---

### 🔹 [3.3 Dynamic Scheduler & Adaptive Timeout](./3.3_adaptive_scheduler)
- **สถานะ:** ฟีเจอร์ที่เราพัฒนาเสริมขึ้นมาเอง (Our Contribution) เพื่อเติมเต็มช่องโหว่ของเปเปอร์ที่ใน GitHub ขาดหายไป
- **เงื่อนไข:** Heavy 3D Augmentation, Max Workers = 8, Batch Size = 2
- **2 คอนฟิกูเรชัน:**
  - `C1`: MinatoLoader + Dynamic WorkerScheduler (ปรับ 2 ➔ 8 workers ตามโหลด CPU)
  - `C2`: MinatoLoader + Dynamic WorkerScheduler + Adaptive Timeout (P75 Rolling Window)
- **ข้อค้นพบสำคัญ (Key Insights):** แม้ระบบปรับ worker ได้จริง แต่ Throughput โดยรวมลดลงเนื่องจาก Overhead ของ Python Multiprocessing & GIL จึงเสนอแนะว่าควรทำ Auto-tune ที่ระดับ C++ Core Engine

---

### 🔹 [3.4 Worker Scaling Sweep (1 to 8 Workers)](./3.4_worker_scaling_sweep)
- **สถานะ:** การทดลองกวาดจำนวน Worker อย่างละเอียดตั้งแต่ 1 ถึง 8 ตัว
- **วัตถุประสงค์:** พิสูจน์จุดเปลี่ยนผัน (Inflection Point) และสาเหตุที่ 8 Workers ทำงานช้ากว่า 4 Workers บนซีพียู Hybrid Architecture (Intel Core i5-12450HX: 4 P-Cores + 4 E-Cores)
- **ผลลัพธ์:** พล็อตกราฟ Scaling Curve ของ Throughput, GPU Starvation และ Hardware Utilization

---

## 🏆 เอกสารสรุปผลและรายงาน
- **[final_thesis_results.md](./final_thesis_results.md):** รายงานผลวิทยานิพนธ์ฉบับสมบูรณ์ กราฟเปรียบเทียบทั้ง 6 คอนฟิก (Throughput, Wait Time, RAM) และบทวิเคราะห์เชิงลึก
- **[advisor_update.md](./advisor_update.md):** บันทึกการอัปเดตความคืบหน้าแก่อาจารย์ที่ปรึกษา
- **`plot_all_configurations.py`:** สคริปต์กลางสำหรับดึงผลลัพธ์จาก 3.2 และ 3.3 มาสร้างกราฟสรุปทั้งหมดใน `./results/plots/new_phase2_thesis/`

---

## 🛠️ ทรัพยากรส่วนกลาง (Shared Resources)
- `.venv_minato/`: Virtual Environment หลักสำหรับรันการทดลอง
- `MinatoLoader/`: ซอร์สโค้ดต้นฉบับจากเปเปอร์ EuroSys 2026
- `scripts/`: สคริปต์ดาวน์โหลดและเตรียมข้อมูล KiTS19
