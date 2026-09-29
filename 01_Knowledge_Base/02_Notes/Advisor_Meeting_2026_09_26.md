---
tags:
  - advisor-meeting
  - research-direction
  - meeting-notes
date: 2026-09-26
---

# 📋 บันทึกการพบอาจารย์ — 26/09/2026

---

## 📌 สรุปผล: RAM-Aware / Rollback = Minor Contribution

อาจารย์ชี้ว่า **Proactive RAM Guard + Safety Rollback ถือเป็น Minor Contribution** เท่านั้น — ต้องหาจุดแข็งหลักที่ใหญ่กว่านี้

---

## 🔭 ทิศทางที่อาจารย์แนะนำให้พิจารณา (3 แนวทาง)

### แนวที่ 1: MinatoLoader ทำ Prefetch Tuning มั้ย?

- **MinatoLoader:** ใช้ P75 Threshold เพื่อสลับ Fast/Slow Queue เท่านั้น → **ไม่ได้ Auto-Tune `prefetch_factor` แบบ Dynamic**
- **โอกาสของเรา:** Dynamic Prefetch Tuning ตาม real-time RAM + Batch Size → เป็น Tuning Parameter ที่ MinatoLoader ไม่ได้จูน

| | MinatoLoader | งานเรา |
|---|---|---|
| `num_workers` auto-tune | ✅ (P75 Heuristic) | ✅ |
| `prefetch_factor` auto-tune | ❌ | ✅ ⭐ |
| Joint tuning ทั้งคู่พร้อมกัน | ❌ | ✅ ⭐ |

---

### แนวที่ 2: Cloud / Multi-Node เป็น Tuning Dimension

- **MinatoLoader:** Local In-Node เท่านั้น — ไม่มี Cross-Node Awareness
- **แนวคิด:** ถ้า Local RAM ตึง → offload prefetch บางส่วนไปยัง node อื่น (CPU Server / Cloud Storage)
- **สถานะ:** น่าสนใจ แต่ Scope ใหญ่มาก
  - **แนะนำ:** Mention เป็น Future Work ในวิทยานิพนธ์ หากมีเวลาจริงๆ ค่อย Implement

---

### แนวที่ 3: Unified Memory Architecture (Mac M-series / Apple Silicon)

- **MinatoLoader ทดสอบบน HPC** (RAM แยก VRAM โดยสมบูรณ์)
- **Unified Memory (Mac M1/M2/M3):** CPU RAM = GPU VRAM → ใช้ Pool เดียวกัน
  - `prefetch_factor` สูง → RAM หมดเร็วกว่า HPC มาก → OOM ง่ายกว่าเดิม
  - งานเรา (Memory Guard) จะ **เด่นมากบนสถาปัตยกรรมนี้** เพราะ Risk OOM สูงกว่า
  - MinatoLoader ยังไม่เคยทดสอบบน Unified Memory → **เราเป็น First**

**Story ที่ขาย:** *"Our Middleware generalizes across architectures — both CUDA (discrete VRAM) and Apple MPS (Unified Memory), where OOM risk is fundamentally higher"*

---

## 🖥️ Hardware Target: ควรใช้อะไร?

| Hardware | จุดเด่น | จุดด้อย | คะแนน |
|---|---|---|---|
| **Mac Mini M1 (16GB Unified)** | Unique case study, ยัง Virgin สำหรับ MinatoLoader, PyTorch MPS backend | Dev env ใหม่ (MPS ≠ CUDA), ต้อง verify MPS support | ⭐⭐⭐⭐ |
| **Linux Workstation (16-32GB CUDA)** | CUDA ecosystem สมบูรณ์, Replicable ง่าย, compatible กับ baseline ทั้งหมด | ธรรมดา ไม่แตกต่างจาก MinatoLoader positioning | ⭐⭐⭐ |
| **Raspberry Pi / ARM Edge** | Exotic | ไม่เหมาะกับ ML Training จริงๆ | ⭐ |

### ✅ แนะนำ: มุ่งไปที่ Mac Mini M1

เหตุผลหลัก:
1. MinatoLoader **ไม่เคย test** บน Unified Memory → เราเป็นคนแรก
2. OOM Risk สูงกว่า HPC → Memory Guard ของเรา **เด่นชัดกว่า**
3. Commodity Hardware ราคาไม่แพง → ตรง Positioning เดิม
4. ถ้า test ได้ทั้ง CUDA (Linux) + MPS (Mac M1) → Contribution แข็งขึ้นมาก

---

## 📐 อัปเดต Research Positioning (หลังประชุม)

```
ก่อนประชุม:
  "Local In-Node, Proactive Memory-Aware, Commodity HW (16-32GB CUDA)"

หลังประชุม (เพิ่ม):
  "Local In-Node, Proactive Memory-Aware, Commodity HW (16-32GB)
   — รองรับทั้ง CUDA Discrete VRAM และ Unified Memory (Apple MPS)"
```

---

## ❓ Open Questions (ต้องตัดสินใจ)

- [ ] จะ Implement Multi-Node Awareness จริงๆ หรือ Future Work เท่านั้น?
- [ ] มี Mac Mini M1 ให้ยืม/ใช้ทดสอบได้จริงหรือเปล่า?
- [ ] MPS Backend ของ PyTorch รองรับ DataLoader multi-worker ได้ Full หรือยัง? (ต้องเช็ค)
- [ ] ถ้าเพิ่ม Unified Memory ใน Contribution → Experiment เพิ่มกี่ชุด?

---

## 🗂️ ไฟล์ที่เกี่ยวข้อง

- [[AutoTuning_Taxonomy_and_Advisor_Feedback]] — Taxonomy ภาพรวมทั้งหมด
- [[MinatoLoader_Citation_and_Foundation_Analysis]] — วิเคราะห์ MinatoLoader โดยละเอียด
