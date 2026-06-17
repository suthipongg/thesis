import pandas as pd
import matplotlib.pyplot as plt
import os

# ==========================================
# ตั้งค่า Theme สำหรับ Presentation + แก้ฟอนต์ภาษาไทยบน Linux
# ==========================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Garuda', 'Loma', 'Kinnari', 'DejaVu Sans'] + plt.rcParams['font.sans-serif']
plt.rcParams['axes.unicode_minus'] = False

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.5,
    'grid.linestyle': '--'
})

def load_detailed_csv(csv_path):
    """
    ฟังก์ชันบังคับอ่านไฟล์ CSV เนื่องจาก Header เดิมมีแค่ 6 คอลัมน์ แต่ข้อมูลจริงมี 10 คอลัมน์
    """
    col_names = [
        'Epoch', 'Batch', 
        'Disk_IO_Sec', 'CPU_Transform_Sec', 'DataLoader_Wait_Sec', 
        'GPU_H2D_Sec', 'GPU_Fwd_Sec', 'GPU_Bwd_Sec', 'GPU_Opt_Sec', 
        'Loss'
    ]
    # ข้ามบรรทัดแรก (skiprows=1) ที่เป็น Header เก่าทิ้งไป แล้วใช้ Header ใหม่แทน
    df = pd.read_csv(csv_path, names=col_names, skiprows=1)
    
    # รวมเวลาฝั่ง GPU เข้าด้วยกันเพื่อความดูง่ายในบางกราฟ
    df['GPU_Total_Sec'] = df['GPU_H2D_Sec'] + df['GPU_Fwd_Sec'] + df['GPU_Bwd_Sec'] + df['GPU_Opt_Sec']
    return df

def plot_detailed_bottleneck(csv_path, output_dir):
    """
    1. กราฟพระเอก (อัปเดตใหม่): โชว์ชำแหละเวลา Disk vs CPU vs GPU แบบละเอียด
    """
    df = load_detailed_csv(csv_path)
    
    # ตัดข้อมูลมาแค่ 40 Batch แรกเพื่อให้กราฟไม่แน่นเกินไป (เห็นพฤติกรรม Cold Start ชัดๆ)
    df_plot = df.head(40).copy()
    x = df_plot['Batch'].astype(str)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # พล็อต Stacked Bar เรียงจากล่างขึ้นบน
    # 1. GPU (สีน้ำเงิน) - เวลาที่ควรจะใช้เยอะที่สุด
    ax.bar(x, df_plot['GPU_Total_Sec'], label='GPU Compute (Fwd+Bwd+Opt)', color='#457b9d')
    
    # 2. CPU Transform (สีส้ม) - เวลาที่ CPU ใช้แปลงรูป
    ax.bar(x, df_plot['CPU_Transform_Sec'], bottom=df_plot['GPU_Total_Sec'], 
           label='CPU Transform (Resize/Augment)', color='#f4a261')
    
    # 3. Disk I/O (สีแดง) - เวลาที่ใช้ดึงไฟล์จากดิสก์ (คอขวดที่แท้จริง)
    ax.bar(x, df_plot['Disk_IO_Sec'], 
           bottom=df_plot['GPU_Total_Sec'] + df_plot['CPU_Transform_Sec'], 
           label='Disk I/O (Read Storage)', color='#e63946')
    
    # 4. DataLoader Wait (สีเทา) - เวลา Overhead ในการส่งข้อมูล
    ax.bar(x, df_plot['DataLoader_Wait_Sec'], 
           bottom=df_plot['GPU_Total_Sec'] + df_plot['CPU_Transform_Sec'] + df_plot['Disk_IO_Sec'], 
           label='DataLoader Overhead', color='#6c757d')
    
    ax.set_title('Detailed Time Breakdown per Batch (First 40 Steps)', fontweight='bold')
    ax.set_xlabel('Batch Step (จำนวนรอบ)')
    ax.set_ylabel('Time (Seconds) / เวลาที่ใช้ (วินาที)')
    
    # หมุนแกน X ให้ตัวเลขไม่ทับกัน
    plt.xticks(rotation=45, fontsize=10)
    # วาง Legend ไว้ข้างนอกกราฟ
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'slide_1_detailed_bottleneck.png'))
    plt.close()
    print("✅ สร้างรูป Slide 1: slide_1_detailed_bottleneck.png เสร็จแล้ว")

def plot_learning_curve(csv_path, output_dir):
    """
    2. กราฟยืนยันความถูกต้อง: โชว์ Loss อย่างเดียว (เพราะเราตัด Accuracy ออกไปในโค้ดใหม่)
    """
    df = load_detailed_csv(csv_path)
    
    df['Loss_Smooth'] = df['Loss'].rolling(window=5, min_periods=1).mean()
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color1 = '#d62828'
    ax1.set_xlabel('Batch Step')
    ax1.set_ylabel('Training Loss', color=color1, fontweight='bold')
    
    ax1.plot(df['Batch'], df['Loss'], color=color1, alpha=0.3, label='Raw Loss')
    ax1.plot(df['Batch'], df['Loss_Smooth'], color=color1, linewidth=2, label='Smoothed Loss')
    
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.legend(loc='upper right')
    
    plt.title('Training Loss Curve (ResNet-50)', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'slide_2_learning_curve.png'))
    plt.close()
    print("✅ สร้างรูป Slide 2: slide_2_learning_curve.png เสร็จแล้ว")

if __name__ == '__main__':
    result_dir = './thesis_results_real'
    train_csv = os.path.join(result_dir, 'training_metrics.csv')
    
    print("กำลังประมวลผลกราฟสำหรับ Presentation...")
    if os.path.exists(train_csv):
        plot_detailed_bottleneck(train_csv, result_dir)
        plot_learning_curve(train_csv, result_dir)
    else:
        print(f"❌ ไม่พบไฟล์ {train_csv}")