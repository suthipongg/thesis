# 3.1 Initial Lightweight Baseline (No Heavy Preprocessing)

การทดลองเฟสแรกเพื่อทดสอบเปรียบเทียบ **PyTorch DataLoader** กับ **MinatoLoader (EuroSys 2026)** ในสภาพแวดล้อมเริ่มต้นแบบปกติ (Light Preprocessing / 2 Workers)

---

## 🎯 วัตถุประสงค์ (Objective)
1. ติดตั้งและจำลองสภาพแวดล้อมโมเดล **3D-UNet** กับชุดข้อมูล **KiTS19**
2. รันเปรียบเทียบ Throughput และ GPU Wait Time ระหว่าง PyTorch Baseline และ MinatoLoader ดั้งเดิม
3. ทดสอบการทำงานของ Multi-process และ Queue

---

## ⚙️ การตั้งค่าการทดลอง (Setup)
- **Dataset:** KiTS19 Subset (3D CT Volumes)
- **Model:** 3D-UNet
- **Workers:** 2 workers
- **Batch Size:** 2 (Patch size: 64×64×32)
- **Augmentation:** แบบเบา (Light Augmentations: Random Flip, Random Brightness)
- **Epochs:** 5 epochs

---

## 📊 ผลการทดลองและข้อสังเกตสำคัญ (Key Findings)
- ผลปรากฏว่า **MinatoLoader ทำความเร็วได้พอๆ กับ หรือช้ากว่า PyTorch เล็กน้อย**
- **สาเหตุที่ค้นพบ:**
  1. **CPU Not Bottleneck:** ด้วย Augmentation แบบเบา CPU สามารถประมวลผลข้อมูลส่งให้ GPU ได้ทันอย่างสบายๆ ทำให้ GPU ไม่เกิดคอขวด
  2. **Fast/Slow Queue ของ Minato ไม่ได้ทำงาน:** เนื่องจากไม่มีตัวอย่างข้อมูลใดใช้เวลาเกิน Timeout (2.0s) ข้อมูลทั้งหมดจึงวิ่งเข้า Fast Queue เพียงอย่างเดียว
  3. **Heavy Preprocessing ขาดหายไป:** โค้ดต้นฉบับใน GitHub ยังไม่ได้เปิดใช้งานตัว Transform หนักๆ เช่น `ElasticDeformation` และ `HeavyGaussianBlur`

> [!NOTE]
> ข้อสังเกตข้างต้นนี้นำไปสู่การออกแบบการทดลองใน **3.2 (Heavy Comparison)** เพื่อจำลองสถานการณ์คอขวดที่แท้จริง

---

## 📁 โครงสร้างโฟลเดอร์
```
3.1_light_baseline/
├── run_pytorch_baseline.py   # สคริปต์รัน PyTorch Baseline (2 Workers)
├── run_minato_baseline.py    # สคริปต์รัน MinatoLoader ดั้งเดิม (2 Workers)
├── instrumented_dataset.py   # Wrapper วัดเวลาการอ่านและ augment รายภาพ
├── plot_results.py           # สคริปต์วาดกราฟเปรียบเทียบ
├── run_light_baseline.sh     # สคริปต์รันทั้งระบบแบบอัตโนมัติ
└── results/
    ├── pytorch/              # Raw logs & metrics ของ PyTorch
    ├── minato/               # Raw logs & metrics ของ MinatoLoader
    └── plots/                # กราฟสรุปผล (Epoch metrics, Timelines, Queue)
```
