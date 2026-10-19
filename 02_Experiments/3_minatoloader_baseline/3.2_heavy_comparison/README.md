# 3.2 Heavy Augmentation Comparison (CPU Bottleneck Benchmark)

การทดลองเปรียบเทียบเชิงลึกระหว่าง **PyTorch DataLoader** และ **MinatoLoader Baseline** ภายใต้สภาวะ **Heavy Preprocessing / Data Augmentations** เต็มรูปแบบ เพื่อจำลองสภาวะ CPU Bottleneck และ Head-of-Line blocking ในการเทรนภาพการแพทย์ 3D

---

## 🎯 วัตถุประสงค์ (Objective)
1. บังคับเปิดใช้งาน Heavy Augmentation ที่ขาดหายไปในโค้ดต้นฉบับ (`ElasticDeformation`, `HeavyGaussianBlur`, `RandFlip`, `GaussianNoise`)
2. เปรียบเทียบประสิทธิภาพระหว่าง PyTorch และ MinatoLoader เมื่อสเกลจำนวน Worker:
   - **4 Workers:** โหลดระดับเหมาะสม (Sweet spot)
   - **8 Workers:** โหลดระดับสูงสุดของ CPU (Stress test / Over-subscription)
3. วัดผลทั้ง **Throughput (img/s)**, **GPU Starvation (Wait Time)**, และ **RAM Usage**

---

## ⚙️ การตั้งค่า 4 คอนฟิกูเรชัน (Configurations)
| Config | ระบบ (System) | จำนวน Worker | Augmentation | Batch Size |
|:---|:---|:---:|:---:|:---:|
| **A1** | PyTorch DataLoader | 4 | Heavy 3D | 2 |
| **A2** | PyTorch DataLoader | 8 | Heavy 3D | 2 |
| **B1** | MinatoLoader (Original) | 4 | Heavy 3D | 2 |
| **B2** | MinatoLoader (Original) | 8 | Heavy 3D | 2 |

---

## 📊 ผลการทดลองและข้อค้นพบ (Key Findings)
- 🏆 **B1 (Minato Fixed 4 Workers) ทำ Throughput ได้สูงสุด (~30 img/s):**
  เนื่องจาก 4 Workers เป็นจุด Sweet spot ของซีพียูเครื่องทดลอง การไม่มี Dynamic Overhead ทำให้ Fast/Slow Queue จัดการงานได้เต็มประสิทธิภาพ
- 📉 **B2 (Minato Fixed 8 Workers) เกิดคอขวด (~21 img/s):**
  เมื่อเปิด 8 Workers เกินขีดจำกัด เกิด Context Switching และ CPU contention ดึง Throughput ตก
- ⏱️ **PyTorch (A1, A2) เผชิญกับ Head-of-Line Blocking:**
  ภาพที่เจอการคำนวณหนักจะขวางแถว ทำให้ GPU มีช่วง Starvation สูงกว่า MinatoLoader

---

## 📁 โครงสร้างโฟลเดอร์
```
3.2_heavy_comparison/
├── run_heavy_experiments.py    # สคริปต์รัน Heavy Augmentation Benchmark (PyTorch / Minato)
├── run_heavy_comparison.sh     # สคริปต์รันรวดเดียว A1, A2, B1, B2 พร้อมวาดกราฟ
├── plot_epoch_metrics.py       # วาดกราฟ 4-Grid เปรียบเทียบทั้ง 4 คอนฟิก
├── plot_phase2_timelines.py    # วาด GPU Execution Timelines ราย Batch
├── plot_epoch5_timelines.py    # วาด Detailed Timelines ของ Epoch สุดท้าย
├── evaluation_cases.txt        # ไฟล์รายชื่อ Case สำหรับแบ่ง Train/Val
└── results/
    ├── heavy_out_A1/           # ผลลัพธ์ A1: PyTorch (4W)
    ├── heavy_out_A2/           # ผลลัพธ์ A2: PyTorch (8W)
    ├── heavy_out_B1/           # ผลลัพธ์ B1: Minato (4W)
    ├── heavy_out_B2/           # ผลลัพธ์ B2: Minato (8W)
    └── plots/                  # กราฟสรุปผล (Metrics & Timelines)
```
