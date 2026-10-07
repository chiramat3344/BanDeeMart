# บ้านดีมาร์ท — วิเคราะห์ยอดขายและประสิทธิภาพช่องทางการขาย

**Baan Dee Mart — Thai E-commerce Sales Analytics**

## ภาพรวมโปรเจกต์

Case Study สำหรับ Data Analyst ระดับ Beginner-to-Junior ที่จำลองธุรกิจร้านค้าออนไลน์ SME ไทย บ้านดีมาร์ทจำหน่ายสินค้าอุปโภคบริโภคและของใช้ในบ้านผ่าน Shopee, Lazada, TikTok Shop, Facebook และ Website เป้าหมายคือเปลี่ยนข้อมูล Order Line ที่ไม่สมบูรณ์ให้เป็นข้อสรุปทางธุรกิจที่เจ้าของร้านใช้วางแผนการขายได้

> **Synthetic Dataset:** ข้อมูลทั้งหมดเป็นข้อมูลจำลองที่สร้างขึ้นเพื่อ Portfolio และการเรียนรู้ ไม่ใช่ข้อมูลจากบริษัทจริง

## ปัญหาและวัตถุประสงค์ทางธุรกิจ

เจ้าของร้านต้องการทราบช่องทางที่สร้างยอดขายและคำสั่งซื้อมากที่สุด สินค้า/หมวดหมู่ที่ควรดูแล ตลาดตามจังหวัดและภูมิภาค พฤติกรรมซื้อซ้ำ และแนวทางจัดสรรเวลาและทรัพยากรให้แต่ละช่องทาง

## Dataset Description

หนึ่งแถวคือสินค้า 1 รายการภายในคำสั่งซื้อ 1 รายการ (Order Line) ดังนั้นต้องนับ `order_id` แบบไม่ซ้ำเมื่อต้องการจำนวนคำสั่งซื้อ Raw และ Cleaned แยกกันชัดเจน ครอบคลุมปี 2024 และมีคอลัมน์ `order_id`, `order_date`, `customer_id`, `product_id`, `product_name`, `category`, `quantity`, `unit_price`, `discount`, `sales_amount`, `province`, `region`, `channel`, `payment_method`, `order_status`

## เครื่องมือและขั้นตอน

Python, pandas, NumPy, Matplotlib, Jupyter และ ReportLab ครอบคลุม Business Understanding, Data Understanding, Data Cleaning, Feature Engineering, Sales Analysis, Customer Analysis, Descriptive Statistics, Visualization และ Business Insight

## Data Cleaning

ตรวจรูปแบบวันที่, duplicate, missing discount, quantity ที่ไม่สมเหตุสมผล, ราคา outlier, status และความสอดคล้องของ `sales_amount` กับ `quantity × unit_price × (1 - discount)` โดยแยกยอดขายของ `Completed` ออกจาก `Cancelled`/`Returned` และบันทึกเหตุผลทุกขั้นตอนในรายงานการทำความสะอาดข้อมูล

## Key Metrics

| Metric | Value |
|---|---:|
| ยอดขายสุทธิ | 1,326,041.22 บาท |
| คำสั่งซื้อ | 1,591 |
| ลูกค้า | 418 |
| จำนวนชิ้นที่ขาย | 3,110 |
| AOV | 833.46 บาท |
| Median Order Value | 622.35 บาท |
| Repeat Purchase Rate | 77.27% |

## Key Findings

- ช่องทางยอดขายสูงสุดคือ **Shopee** (363,080.40 บาท)
- หมวดหมู่ยอดขายสูงสุดคือ **ของใช้ในบ้าน** (431,876.17 บาท)
- เดือนยอดขายสูงสุดคือเดือนที่ 12 (148,761.58 บาท) และต่ำสุดคือเดือนที่ 6 (77,614.37 บาท)
- จังหวัดสำคัญที่สุดคือ **สมุทรปราการ** (109,033.55 บาท)
- ลูกค้าที่ซื้อซ้ำคิดเป็น **77.27%** ของลูกค้าในช่วงข้อมูลนี้
- สินค้า 15 จาก 24 รายการสะสมยอดขายได้ 80% ของยอดขายทั้งหมด

## Business Recommendations

1. จัดสรรเวลาติดตามและงบส่งเสริมการขายให้ **Shopee** เป็นช่องทางหลัก แต่ติดตาม AOV และจำนวนคำสั่งซื้อแยกกันก่อนเพิ่มงบ
2. รักษาสต็อก **ของใช้ในบ้าน** และทดสอบ Bundle กับสินค้ายอดขายต่ำ โดยตรวจสอบกำไรจริงก่อนตัดสินใจ
3. ใช้แผนรายเดือนตามรูปแบบที่พบ และเก็บข้อมูลหลายปีเพื่อยืนยันฤดูกาล
4. ทำ CRM สำหรับลูกค้าที่ซื้อซ้ำ และทำแคมเปญดึงลูกค้าใหม่โดยวัด Conversion เพิ่มเติม
5. ใช้จังหวัด/ภูมิภาคเป็นจุดตั้งต้นในการวางแผนส่งสินค้า ไม่สรุปสาเหตุเชิงการตลาดจนกว่าจะมีข้อมูลค่าโฆษณาและ Traffic

## ข้อจำกัด

เป็น Synthetic Data ปีเดียว ไม่มีต้นทุน กำไร ค่าโฆษณา Traffic และข้อมูลลูกค้าก่อนช่วงวิเคราะห์ จึงรายงานเป็นยอดขาย/รายได้ ไม่ใช่กำไร และ Repeat Customer หมายถึงซื้อซ้ำภายในช่วงข้อมูลเท่านั้น

## โครงสร้างโปรเจกต์และวิธีใช้งาน

- `data/raw/orders_raw.csv` ข้อมูลดิบ
- `data/cleaned/orders_cleaned.csv` ข้อมูลหลังทำความสะอาด
- `notebooks/baan_dee_mart_analysis.ipynb` Notebook ภาษาไทย
- `outputs/` กราฟ 10 รายการ
- `reports/` Markdown และ PDF สำหรับบุคคลภายนอก

```bash
pip install -r requirements.txt
python scripts/build_baan_dee_mart.py
jupyter notebook notebooks/baan_dee_mart_analysis.ipynb
```

## ทักษะที่แสดงให้เห็น

Data Cleaning, pandas, Order-level Metrics, Sales Analysis, Customer Analysis, Descriptive Statistics, Visualization, Business Insight, Business Recommendation และการเขียนรายงานสำหรับ SME
