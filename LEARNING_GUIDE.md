# Learning Guide — บ้านดีมาร์ท

คู่มือทบทวนสำหรับโปรเจกต์นี้ เน้นสิ่งที่ Data Analyst ระดับ Beginner-to-Junior ต้องอธิบายได้

| หัวข้อ | ต้องเข้าใจและใช้ตรงไหน | คำถามฝึกคิด |
|---|---|---|
| Python Basics | ตัวแปร ฟังก์ชัน loop และการอ่านไฟล์ ใช้ใน pipeline | ทำไมแยกฟังก์ชันสร้าง/clean/metric? |
| Pandas | DataFrame, `loc`, `isna`, `drop_duplicates` ใช้ตรวจและแก้ข้อมูล | `count` กับ `nunique` ต่างกันอย่างไร? |
| Data Types/Datetime | แปลง `order_date` และใช้ `.dt` ทำรายเดือน | ถ้าวันที่ parse ไม่ได้ควรทำอย่างไร? |
| Missing Values | แยก missing discount ที่ยืนยันได้จาก missing ที่ไม่รู้ | ทำไมเติม 0 ทุกแถวไม่ได้? |
| Duplicate/Outlier | duplicate ทั้งแถวและราคาเทียบ Median สินค้า | outlier ทุกค่าคือข้อมูลผิดหรือไม่? |
| GroupBy/Aggregation | รวมยอดขายและนับ order/customer ที่ระดับถูกต้อง | ทำไม AOV ต้องรวมต่อ order ก่อน? |
| Feature Engineering | `gross_sales`, `net_sales`, เดือน, customer type และ bucket | ฟีเจอร์นี้ตอบ Business Question ข้อใด? |
| Descriptive Statistics | Mean, Median, SD, Percentile และ Distribution | outlier มีผลต่อ Mean อย่างไร? |
| Visualization | เลือก line/bar/histogram ให้ตอบคำถาม | กราฟนี้ช่วยตัดสินใจอะไร? |
| Business Metrics | Sales, Orders, Customers, AOV, Repeat Rate | ยอดขายสูงหมายถึงกำไรสูงหรือไม่? |
| Customer Analysis/RFM | Frequency และ Monetary; Recency ต้องมี snapshot ที่ชัด | Repeat ในปีเดียวตีความได้แค่ไหน? |
| Correlation | ความสัมพันธ์ไม่ใช่เหตุและผล | ต้องมีข้อมูลอะไรจึงอธิบายสาเหตุได้? |
| Insight/Storytelling | Finding → Evidence → Implication → Recommendation → Limitation | ข้อเสนอแนะผูกกับตัวเลขใด? |

ควรอ่านเพิ่มเติม: pandas User Guide, Matplotlib fundamentals, SQL aggregation, Business Metrics สำหรับ E-commerce และหลักการ Data Storytelling
