# 🗺️ Thesis Execution Roadmap
*(แผนปฏิบัติการ: ขั้นตอนการทำงานวิจัยเพื่อไปสู่เป้าหมาย)*

เพื่อเปลี่ยน **"ไอเดีย"** ให้กลายเป็น **"วิทยานิพนธ์ที่สำเร็จ"** เราจะแบ่งการทำงานออกเป็น 4 ระยะ (Phases) ดังนี้:

---

## Phase 1: 🏗️ Baseline Benchmarking & Proof of Concept
**เป้าหมาย:** สร้างสภาพแวดล้อมเพื่อจำลองปัญหา และวัดผลตัวเปรียบเทียบ (Baseline)
1. **เลือก Dataset & Model:** 
   - เลือก Dataset ขนาดใหญ่และมีการทำ Augmentation หนักๆ เช่น *KiTS19 (3D Medical Image)* หรือ *COCO (Object Detection)*
2. **สร้าง Baseline Script:**
   - เขียนโค้ด PyTorch Training Loop แบบมาตรฐาน
   - ใช้ PyTorch DataLoader แบบปกติ (`num_workers` คงที่)
3. **ทดสอบเพื่อสร้าง "ปัญหา" ให้เห็นชัดเจน:**
   - ทดสอบรันเพื่อหาจุดที่เกิดการคอขวด (GPU ว่างงานรอข้อมูล)
   - ดันค่า `num_workers` ขึ้นจนกระทั่งระบบเกิด **Out of Memory (OOM)** หรือเครื่องค้าง
   - บันทึกตัวเลข: *GPU Utilization (%), Training Time per Epoch, Peak RAM Usage* 

## Phase 2: 🧠 Core System Development (ระบบหลัก)
**เป้าหมาย:** สร้าง Middleware ตามโครงร่าง 3 เสาหลัก
1. **โมดูล Memory-Aware Profiler:**
   - เขียนสคริปต์สกัด Resource (RAM/CPU) ที่ใช้ต่อ 1 Sample (Warm-up phase)
   - สร้างฟังก์ชันเช็ค Memory เพดาน (เช่น RAM ห้ามเกิน 85%) 
2. **โมดูล Dynamic Auto-Tuner:**
   - อิมพลีเมนต์สมการคิวทฤษฎี $\omega = t_{fetch} / t_{req}$ เพื่อคอยจับเวลาว่า Data Loading ช้ากว่า GPU หรือไม่
   - เขียนฟังก์ชันให้ PyTorch เปลี่ยน `num_workers` หรือ `prefetch_factor` กลาง Epoch ได้ (ผ่าน Iterator wrapper หรือ Background Thread)
3. **โมดูล Safety Rollback:**
   - สร้างตัวแปรเก็บ `best_throughput`
   - ทำเงื่อนไข: หากปรับจูนแล้ว 50 Steps พบว่า Throughput ตก ให้ Revert ค่ากลับทันที

## Phase 3: 🧪 Integration & Evaluation (ทดสอบและเทียบผล)
**เป้าหมาย:** พิสูจน์ว่าระบบของเราดีกว่า SOTA และใช้งานได้จริง
1. **Drop-in Test:**
   - นำ Middleware ไปครอบ DataLoader ตัวเดิมในสคริปต์ที่เขียนไว้ใน Phase 1
2. **Performance Comparison:**
   - เทียบกับ **PyTorch Default** (Baseline)
   - เทียบกับ **PyTorch ที่ปรับจูนโดยมนุษย์** (Manual Tuning ที่ดีที่สุด)
   - *(ถ้าเป็นไปได้)* เทียบกับโค้ดของเปเปอร์อื่น เช่น **MinatoLoader**
3. **Metric ที่ต้องวัด:**
   - 📈 GPU Utilization สูงขึ้นกี่ %
   - ⏱️ เวลาเทรนรวม (Total Time to Train) ลดลงกี่ %
   - 🛡️ อัตราการรอดพ้นจาก OOM (Crash Rate) บนเครื่อง RAM จำกัด

## Phase 4: 📝 Thesis Writing & Defense
**เป้าหมาย:** สรุปผลการทดลองและเขียนเล่มวิทยานิพนธ์
- **บทที่ 1:** บทนำ (หยิบจาก Presentation Script)
- **บทที่ 2:** ทบทวนวรรณกรรม (สรุป Gaps จากเปเปอร์ MinatoLoader, DLCache, Plumber ฯลฯ)
- **บทที่ 3:** ระเบียบวิธีวิจัย (อธิบายสถาปัตยกรรม Memory-Aware + Rollback)
- **บทที่ 4:** ผลการทดลอง (นำกราฟจาก Phase 3 มาวิเคราะห์เชิงลึก)
- **บทที่ 5:** สรุปผลและข้อเสนอแนะ
