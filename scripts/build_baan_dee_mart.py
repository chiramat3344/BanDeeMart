"""สร้าง Dataset, Notebook, กราฟ และรายงานภาษาไทยของบ้านดีมาร์ทแบบทำซ้ำได้"""

from pathlib import Path
import shutil

import nbformat as nbf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.platypus import PageBreak

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
CLEAN_DIR = ROOT / "data" / "cleaned"
FIG_DIR = ROOT / "outputs"
REPORT_DIR = ROOT / "reports"
NOTEBOOK = ROOT / "notebooks" / "baan_dee_mart_analysis.ipynb"
RAW = RAW_DIR / "orders_raw.csv"
CLEAN = CLEAN_DIR / "orders_cleaned.csv"
SEED = 42
FONT_PATH = Path("C:/Windows/Fonts/tahoma.ttf")
FONT_BOLD_PATH = Path("C:/Windows/Fonts/tahomabd.ttf")

CHANNELS = ["Shopee", "Lazada", "TikTok Shop", "Facebook", "Website"]
PAYMENTS = ["โอนผ่านธนาคาร", "บัตรเครดิต", "เก็บเงินปลายทาง", "e-Wallet"]
REGIONS = ["กรุงเทพฯและปริมณฑล", "ภาคกลาง", "ภาคเหนือ", "ภาคตะวันออกเฉียงเหนือ", "ภาคตะวันออก", "ภาคตะวันตก", "ภาคใต้"]
PROVINCES = {
    "กรุงเทพฯและปริมณฑล": ["กรุงเทพมหานคร", "นนทบุรี", "ปทุมธานี", "สมุทรปราการ"],
    "ภาคกลาง": ["พระนครศรีอยุธยา", "นครปฐม", "สุพรรณบุรี", "สระบุรี"],
    "ภาคเหนือ": ["เชียงใหม่", "เชียงราย", "พิษณุโลก", "ลำปาง"],
    "ภาคตะวันออกเฉียงเหนือ": ["ขอนแก่น", "นครราชสีมา", "อุดรธานี", "อุบลราชธานี"],
    "ภาคตะวันออก": ["ชลบุรี", "ระยอง", "จันทบุรี", "ฉะเชิงเทรา"],
    "ภาคตะวันตก": ["กาญจนบุรี", "ราชบุรี", "เพชรบุรี", "ประจวบคีรีขันธ์"],
    "ภาคใต้": ["สงขลา", "สุราษฎร์ธานี", "ภูเก็ต", "นครศรีธรรมราช"],
}
PRODUCTS = [
    ("P001", "หม้อทอดไร้น้ำมัน 4 ลิตร", "เครื่องใช้ไฟฟ้า", 1690),
    ("P002", "กระทะเคลือบหินอ่อน", "เครื่องครัว", 690),
    ("P003", "กล่องถนอมอาหาร 5 ชิ้น", "เครื่องครัว", 329),
    ("P004", "แก้วน้ำเก็บอุณหภูมิ", "ของใช้ในบ้าน", 459),
    ("P005", "ผ้าขนหนูไมโครไฟเบอร์", "ของใช้ในบ้าน", 199),
    ("P006", "น้ำยาซักผ้าเข้มข้น", "ของใช้ในบ้าน", 289),
    ("P007", "แชมพูสมุนไพร", "ของใช้ส่วนตัว", 219),
    ("P008", "ครีมกันแดด SPF50", "ของใช้ส่วนตัว", 349),
    ("P009", "เซ็ตกล่องข้าว", "เครื่องครัว", 399),
    ("P010", "ชั้นวางของอเนกประสงค์", "ของใช้ในบ้าน", 799),
    ("P011", "หมอนเพื่อสุขภาพ", "ของใช้ในบ้าน", 590),
    ("P012", "ไฟอ่านหนังสือ LED", "เครื่องใช้ไฟฟ้า", 299),
    ("P013", "พัดลมตั้งโต๊ะ USB", "เครื่องใช้ไฟฟ้า", 499),
    ("P014", "กระเป๋าผ้าอเนกประสงค์", "แฟชั่นและไลฟ์สไตล์", 259),
    ("P015", "ขวดน้ำพกพา", "แฟชั่นและไลฟ์สไตล์", 179),
    ("P016", "พรมเช็ดเท้า", "ของใช้ในบ้าน", 249),
    ("P017", "ชุดไม้แขวนเสื้อ", "ของใช้ในบ้าน", 159),
    ("P018", "เครื่องชั่งน้ำหนักดิจิทัล", "สุขภาพ", 649),
    ("P019", "ยางยืดออกกำลังกาย", "สุขภาพ", 229),
    ("P020", "แก้วกาแฟเซรามิก", "ของใช้ในบ้าน", 189),
    ("P021", "กล่องเก็บรองเท้า", "ของใช้ในบ้าน", 279),
    ("P022", "เซ็ตอุปกรณ์ทำความสะอาด", "ของใช้ในบ้าน", 369),
    ("P023", "ปลั๊กไฟ USB", "เครื่องใช้ไฟฟ้า", 429),
    ("P024", "กระเป๋าเก็บอุณหภูมิ", "แฟชั่นและไลฟ์สไตล์", 499),
]
PRODUCT_LOOKUP = {p[0]: p for p in PRODUCTS}


def make_raw():
    rng = np.random.default_rng(SEED)
    n_orders = 1800
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    weights = np.ones(len(dates))
    weights[[d.month in (3, 4, 11, 12) for d in dates]] *= 1.35
    weights = weights / weights.sum()
    customers = [f"CUST-{i:05d}" for i in range(1, 501)]
    customer_weights = rng.gamma(1.5, 1, len(customers)); customer_weights /= customer_weights.sum()
    rows = []
    for i in range(n_orders):
        order_id = f"ORD-{i + 1:06d}"
        date = pd.Timestamp(rng.choice(dates, p=weights))
        region = rng.choice(REGIONS, p=[.25, .16, .13, .15, .11, .07, .13])
        province = rng.choice(PROVINCES[region])
        channel = rng.choice(CHANNELS, p=[.28, .18, .18, .14, .22])
        status = rng.choice(["Completed", "Cancelled", "Returned"], p=[.91, .06, .03])
        customer = rng.choice(customers, p=customer_weights)
        for product_id in rng.choice([p[0] for p in PRODUCTS], size=int(rng.choice([1, 1, 1, 2, 2, 3], p=[.32, .18, .10, .20, .15, .05])), replace=False):
            _, name, category, base = PRODUCT_LOOKUP[product_id]
            quantity = int(rng.choice([1, 1, 1, 2, 2, 3, 4], p=[.40, .22, .10, .13, .08, .05, .02]))
            unit_price = round(base * rng.uniform(.94, 1.06), 2)
            discount = float(rng.choice([0, .05, .10, .15, .20], p=[.55, .20, .15, .08, .02]))
            amount = round(quantity * unit_price * (1 - discount), 2)
            rows.append({"order_id": order_id, "order_date": date.strftime("%Y-%m-%d"), "customer_id": customer, "product_id": product_id, "product_name": name, "category": category, "quantity": quantity, "unit_price": unit_price, "discount": discount, "sales_amount": amount, "province": province, "region": region, "channel": channel, "payment_method": rng.choice(PAYMENTS), "order_status": status})
    df = pd.DataFrame(rows)
    n = len(df)
    # ตั้งใจใส่ปัญหาข้อมูลแบบพอเหมาะสำหรับการฝึกวิเคราะห์
    for idx in rng.choice(n, 16, replace=False): df.loc[idx, "order_date"] = "ไม่ทราบวันที่"
    for idx in rng.choice(n, 75, replace=False): df.loc[idx, "discount"] = np.nan
    for idx in rng.choice(n, 12, replace=False): df.loc[idx, "quantity"] = rng.choice([0, -1])
    for idx in rng.choice(n, 10, replace=False): df.loc[idx, "unit_price"] *= 10
    for idx in rng.choice(n, 70, replace=False): df.loc[idx, "sales_amount"] = (df.loc[idx, "sales_amount"] * rng.uniform(.55, 1.45)).round(2)
    for idx in rng.choice(n, 12, replace=False): df.loc[idx, "order_status"] = np.nan
    channel_idx = rng.choice(n, 25, replace=False)
    df.loc[channel_idx, "channel"] = df.loc[channel_idx, "channel"].str.lower()
    df = pd.concat([df, df.sample(28, random_state=SEED)], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True); df.to_csv(RAW, index=False)
    return df


def clean_data(raw):
    df = raw.copy(); audit = {}
    audit["raw_rows"] = len(df)
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    audit["unparseable_dates"] = int(df["order_date"].isna().sum()); df = df.dropna(subset=["order_date"]).copy()
    audit["duplicates"] = int(df.duplicated().sum()); df = df.drop_duplicates().copy()
    for col in ["channel", "province", "region", "payment_method"]: df[col] = df[col].astype("string").str.strip()
    channel_map = {x.lower(): x for x in CHANNELS}; df["channel"] = df["channel"].str.lower().map(channel_map).fillna(df["channel"])
    df["order_status"] = df["order_status"].fillna("Unknown")
    audit["missing_discount"] = int(df["discount"].isna().sum())
    discount_safe = df["discount"].isna() & ((df["sales_amount"] - df["quantity"] * df["unit_price"]).abs() <= 1)
    audit["discount_filled"] = int(discount_safe.sum())
    audit["discount_unresolved"] = audit["missing_discount"] - audit["discount_filled"]
    df.loc[discount_safe, "discount"] = 0.0
    df = df.dropna(subset=["discount"]).copy()
    audit["invalid_quantity"] = int((df["quantity"] <= 0).sum()); df = df[df["quantity"] > 0].copy()
    medians = df.groupby("product_id")["unit_price"].transform("median"); price_outlier = (df["unit_price"] > medians * 5) | (df["unit_price"] < medians * .2)
    audit["price_outliers"] = int(price_outlier.sum()); df = df[~price_outlier].copy()
    audit["cancelled_or_returned"] = int((df["order_status"] != "Completed").sum()); df = df[df["order_status"] == "Completed"].copy()
    expected = df["quantity"] * df["unit_price"] * (1 - df["discount"]); audit["sales_mismatches"] = int((df["sales_amount"] - expected).abs().gt(1).sum()); df["gross_sales"] = (df["quantity"] * df["unit_price"]).round(2); df["net_sales"] = expected.round(2); df["sales_amount"] = df["net_sales"]
    df["order_month"] = df["order_date"].dt.month; df["order_year"] = df["order_date"].dt.year; df["order_month_name"] = df["order_date"].dt.strftime("%Y-%m")
    orders = df.groupby("order_id", as_index=False).agg(customer_id=("customer_id", "first"), order_date=("order_date", "first"), order_value=("net_sales", "sum"))
    order_counts = orders.groupby("customer_id")["order_id"].nunique(); df["customer_order_count"] = df["customer_id"].map(order_counts); df["customer_type"] = np.where(df["customer_order_count"] > 1, "ลูกค้าที่ซื้อซ้ำ", "ลูกค้าใหม่"); df["order_value_bucket"] = pd.cut(df["customer_id"].map(orders.groupby("customer_id").order_value.sum()), bins=[-np.inf, 1000, 3000, np.inf], labels=["มูลค่าต่ำ", "มูลค่ากลาง", "มูลค่าสูง"])
    CLEAN_DIR.mkdir(parents=True, exist_ok=True); df.to_csv(CLEAN, index=False, date_format="%Y-%m-%d")
    audit["clean_rows"] = len(df); audit["missing_after"] = int(df.isna().sum().sum()); audit["duplicates_after"] = int(df.duplicated().sum())
    return df, audit


def metrics(df):
    orders = df.groupby("order_id", as_index=False).agg(order_date=("order_date", "first"), customer_id=("customer_id", "first"), order_value=("net_sales", "sum"), channel=("channel", "first"))
    customer = orders.groupby("customer_id").agg(order_count=("order_id", "nunique"), sales=("order_value", "sum"))
    product = df.groupby(["product_id", "product_name"], as_index=False).net_sales.sum().sort_values("net_sales", ascending=False)
    cumulative = product.net_sales.cumsum() / product.net_sales.sum(); n80 = int((cumulative < .8).sum() + 1)
    channel_sales = df.groupby("channel").net_sales.sum().sort_values(ascending=False)
    channel_orders = orders.groupby("channel").order_id.nunique().sort_values(ascending=False)
    channel_aov = orders.groupby("channel").order_value.mean().sort_values(ascending=False)
    return {"df": df, "orders": orders, "sales": df.net_sales.sum(), "order_count": len(orders), "customers": df.customer_id.nunique(), "products": df.product_id.nunique(), "quantity": int(df.quantity.sum()), "aov": orders.order_value.mean(), "median": orders.order_value.median(), "std": orders.order_value.std(), "p75": orders.order_value.quantile(.75), "customer": customer, "new": int((customer.order_count == 1).sum()), "repeat": int((customer.order_count > 1).sum()), "repeat_rate": (customer.order_count > 1).mean() * 100, "month": df.groupby("order_month").net_sales.sum(), "channel_sales": channel_sales, "channel_orders": channel_orders, "channel_aov": channel_aov, "category": df.groupby("category").net_sales.sum().sort_values(ascending=False), "product": product, "province": df.groupby("province").net_sales.sum().sort_values(ascending=False), "region": df.groupby("region").net_sales.sum().sort_values(ascending=False), "n80": n80, "top10_share": customer.sales.nlargest(10).sum() / df.net_sales.sum() * 100}


def money(x): return f"{x:,.2f} บาท"
def pc(x): return f"{x:.2f}%"
def write_md(path, text): path.write_text(text.strip() + "\n", encoding="utf-8")


def build_docs(m, audit):
    for old_name in ["BUSINESS_INSIGHTS.md", "DATA_CLEANING_REPORT.md", "EXECUTIVE_SUMMARY.md", "BrightCart_Portfolio_Case_Study.md", "BrightCart_Executive_Summary.md", "BrightCart_Data_Cleaning_Report.md", "BrightCart_Portfolio_Case_Study.pdf", "BrightCart_Executive_Summary.pdf", "BrightCart_Data_Cleaning_Report.pdf"]:
        old_path = REPORT_DIR / old_name
        if old_path.exists():
            old_path.unlink()
    top_channel = m["channel_sales"].index[0]; top_category = m["category"].index[0]; top_province = m["province"].index[0]; best_month = m["month"].idxmax(); low_month = m["month"].idxmin()
    metrics_table = f"| ยอดขายสุทธิ | {money(m['sales'])} |\n| คำสั่งซื้อ | {m['order_count']:,} |\n| ลูกค้า | {m['customers']:,} |\n| จำนวนชิ้นที่ขาย | {m['quantity']:,} |\n| AOV | {money(m['aov'])} |\n| Median Order Value | {money(m['median'])} |\n| Repeat Purchase Rate | {pc(m['repeat_rate'])} |"
    write_md(ROOT / "README.md", f"""# บ้านดีมาร์ท — วิเคราะห์ยอดขายและประสิทธิภาพช่องทางการขาย

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

| Metric | Value |\n|---|---:|\n{metrics_table}

## Key Findings

- ช่องทางยอดขายสูงสุดคือ **{top_channel}** ({money(m['channel_sales'].iloc[0])})
- หมวดหมู่ยอดขายสูงสุดคือ **{top_category}** ({money(m['category'].iloc[0])})
- เดือนยอดขายสูงสุดคือเดือนที่ {best_month} ({money(m['month'].loc[best_month])}) และต่ำสุดคือเดือนที่ {low_month} ({money(m['month'].loc[low_month])})
- จังหวัดสำคัญที่สุดคือ **{top_province}** ({money(m['province'].iloc[0])})
- ลูกค้าที่ซื้อซ้ำคิดเป็น **{pc(m['repeat_rate'])}** ของลูกค้าในช่วงข้อมูลนี้
- สินค้า {m['n80']} จาก {m['products']} รายการสะสมยอดขายได้ 80% ของยอดขายทั้งหมด

## Business Recommendations

1. จัดสรรเวลาติดตามและงบส่งเสริมการขายให้ **{top_channel}** เป็นช่องทางหลัก แต่ติดตาม AOV และจำนวนคำสั่งซื้อแยกกันก่อนเพิ่มงบ
2. รักษาสต็อก **{top_category}** และทดสอบ Bundle กับสินค้ายอดขายต่ำ โดยตรวจสอบกำไรจริงก่อนตัดสินใจ
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
""")
    write_md(ROOT / "PORTFOLIO_CASE_STUDY.md", f"""# บ้านดีมาร์ท — Portfolio Case Study

## Project Snapshot

| รายการ | รายละเอียด |
|---|---|
| ธุรกิจ | ร้านค้าออนไลน์ SME ในประเทศไทย |
| ช่วงเวลา | มกราคม-ธันวาคม 2024 |
| ข้อมูล | Synthetic Dataset สำหรับ Portfolio |
| Grain | หนึ่งแถว = สินค้า 1 รายการในคำสั่งซื้อ 1 รายการ |
| เครื่องมือ | Python, pandas, NumPy, Matplotlib, Jupyter |

## ภาพรวมธุรกิจและ Business Challenge

บ้านดีมาร์ทขายสินค้าออนไลน์ผ่าน 5 ช่องทาง แต่เจ้าของร้านยังไม่มีภาพรวมที่เชื่อถือได้ว่าช่องทางใดสร้างยอดขาย/คำสั่งซื้อสูง สินค้าใดควรดูแล ตลาดจังหวัดใดสำคัญ และลูกค้ากลับมาซื้อซ้ำหรือไม่ โจทย์คือสร้างรายงานที่ใช้ตัดสินใจได้โดยไม่กล่าวอ้างเกินข้อมูล

## Business Objective และคำถาม

วิเคราะห์ยอดขาย ช่องทาง หมวดหมู่ สินค้า จังหวัด ภูมิภาค ลูกค้า และการกระจุกตัวของรายได้ พร้อมเปลี่ยนผลลัพธ์เป็นคำแนะนำที่ตรวจสอบได้ คำถามทั้งหมดคำนวณจาก `order_date`, `order_id`, `customer_id`, `product_id`, `sales_amount`, `province`, `region`, `channel`, `order_status` และฟิลด์ราคาที่เกี่ยวข้อง

## การเตรียมข้อมูล

Raw มี {audit['raw_rows']:,} แถว หลัง parse วันที่ ตัด duplicate ตรวจ quantity/ราคา แยก `Completed` และคำนวณ `net_sales` จากสูตร เหลือ {audit['clean_rows']:,} แถว การเติมส่วนลด 0% ทำเฉพาะแถวที่ตรวจแล้วว่ายอดเดิมตรงกับราคาก่อนส่วนลด ส่วนแถวที่ไม่สามารถยืนยันได้จะไม่ถูกใช้เป็นยอดขาย Completed

## ตัวชี้วัดสำคัญ

| Metric | Value |
|---|---:|
{metrics_table}

## ผลการวิเคราะห์และ Key Findings

1. **ช่องทาง:** {top_channel} ทำยอดขายสูงสุด {money(m['channel_sales'].iloc[0])}; ต้องดูจำนวนออเดอร์และ AOV ร่วมกัน ไม่ใช้ยอดขายอย่างเดียว
2. **เวลา:** เดือน {best_month} สูงสุด {money(m['month'].loc[best_month])} และเดือน {low_month} ต่ำสุด {money(m['month'].loc[low_month])}; ข้อมูลปีเดียวไม่ยืนยันฤดูกาลระยะยาว
3. **สินค้า:** {top_category} เป็นหมวดหมู่หลัก ขณะที่สินค้า {m['n80']} จาก {m['products']} รายการสร้าง 80% ของยอดขาย แสดงการกระจุกตัวระดับหนึ่ง
4. **พื้นที่:** {top_province} เป็นจังหวัดยอดขายสูงสุด {money(m['province'].iloc[0])}; เป็นหลักฐานเพื่อจัดลำดับการติดตาม ไม่ใช่คำอธิบายสาเหตุ
5. **ลูกค้า:** ลูกค้าซื้อซ้ำ {m['repeat']:,} รายจาก {m['customers']:,} ราย หรือ {pc(m['repeat_rate'])}; สถานะนี้จำกัดอยู่ในช่วงข้อมูลปี 2024
6. **มูลค่าตะกร้า:** ค่าเฉลี่ย {money(m['aov'])} เทียบ Median {money(m['median'])}; หากต่างกันมากควรใช้ Median ประกอบการตั้ง threshold

## Business Recommendations

โฟกัสการปฏิบัติงานที่ {top_channel}, วางแผนสต็อกตามหมวด {top_category}, ทำ Bundle/CRM อย่างทดสอบได้, ใช้กลุ่มลูกค้าซื้อซ้ำเป็นฐาน retention และเพิ่มข้อมูลต้นทุน ค่าโฆษณา Traffic และ Margin ก่อนตัดสินใจลงทุนหรือสรุป ROI

## Limitations และสิ่งที่โปรเจกต์แสดงให้เห็น

ข้อมูลจำลองและมีเพียงหนึ่งปี ไม่ใช่ผลประกอบการจริง ไม่มีต้นทุน/กำไรและ attribution การวิเคราะห์จึงเป็น Descriptive Analytics เท่านั้น โปรเจกต์นี้แสดงการเข้าใจ Grain, ทำความสะอาดแบบมีเหตุผล, คำนวณ Order-level Metrics, สื่อสารภาษาไทย และเชื่อม Insight กับการตัดสินใจ SME
""")
    write_md(ROOT / "EXECUTIVE_SUMMARY.md", f"""# บทสรุปผู้บริหาร — บ้านดีมาร์ท

**หมายเหตุ:** รายงานนี้ใช้ **Synthetic Dataset** เพื่อการเรียนรู้และ Portfolio ไม่ใช่ข้อมูลจากบริษัทจริง

## วัตถุประสงค์

สรุปยอดขาย ประสิทธิภาพช่องทาง สินค้า พื้นที่ และพฤติกรรมลูกค้า เพื่อช่วยจัดลำดับความสำคัญในการดูแลร้านค้าออนไลน์

## Key Metrics

| ตัวชี้วัด | ผลลัพธ์ |
|---|---:|
{metrics_table}

## Key Findings

- **{top_channel}** สร้างยอดขายสูงสุด {money(m['channel_sales'].iloc[0])}
- **{top_category}** เป็นหมวดหมู่ยอดขายสูงสุด {money(m['category'].iloc[0])}
- เดือน {best_month} สูงสุด และเดือน {low_month} ต่ำสุด
- **{top_province}** เป็นตลาดจังหวัดที่มียอดขายสูงสุด
- Repeat Purchase Rate อยู่ที่ **{pc(m['repeat_rate'])}**

## ข้อเสนอแนะ

1. จัดลำดับการดูแล {top_channel} พร้อมติดตามคำสั่งซื้อและ AOV แยกกัน
2. รักษาสต็อกหมวด {top_category} และทดสอบ Bundle สินค้าที่ยอดขายต่ำ
3. วางแผนสต็อกและแคมเปญตามเดือน แต่เก็บข้อมูลเพิ่มเพื่อยืนยันฤดูกาล
4. ทำ CRM กับลูกค้าซื้อซ้ำและเพิ่มการวัดผลลูกค้าใหม่

## ข้อจำกัด

ผลลัพธ์เป็นยอดขายจากข้อมูลจำลองปีเดียว ไม่ใช่กำไรหรือ ROI ไม่มีต้นทุน ค่าโฆษณา Traffic และประวัติลูกค้านอกช่วงข้อมูล
""")
    write_md(ROOT / "DATA_CLEANING_REPORT.md", f"""# รายงานการทำความสะอาดข้อมูล — บ้านดีมาร์ท

ข้อมูลนี้เป็น **Synthetic Dataset** ที่สร้างเพื่อ Portfolio โดยตั้งใจใส่ปัญหาข้อมูลในระดับที่สมจริง

## สรุปผล

| รายการ | จำนวน |
|---|---:|
| แถว Raw | {audit['raw_rows']:,} |
| แถว Cleaned | {audit['clean_rows']:,} |
| วันที่อ่านไม่ได้และถูกตัด | {audit['unparseable_dates']:,} |
| Duplicate ที่ลบ | {audit['duplicates']:,} |
| Missing discount ที่พบ | {audit['missing_discount']:,} |
| Missing discount ที่เติมเป็น 0 ได้อย่างมีหลักฐาน | {audit['discount_filled']:,} |
| Missing discount ที่ยืนยันไม่ได้และไม่ใช้ | {audit['discount_unresolved']:,} |
| Quantity ไม่ถูกต้องและถูกตัด | {audit['invalid_quantity']:,} |
| ราคา outlier และถูกตัด | {audit['price_outliers']:,} |
| Cancelled/Returned/Unknown ที่ไม่รวมยอดขาย | {audit['cancelled_or_returned']:,} |
| Sales mismatch ที่คำนวณใหม่ | {audit['sales_mismatches']:,} |
| Missing หลัง cleaning | {audit['missing_after']:,} |
| Duplicate หลัง cleaning | {audit['duplicates_after']:,} |

## หลักการตัดสินใจ

- วันที่อ่านไม่ได้ไม่สามารถใช้วิเคราะห์รายเดือนได้ จึงตัดและบันทึกจำนวน
- Duplicate ทั้งแถวถือเป็นการส่งข้อมูลซ้ำ จึงเก็บรายการแรก
- Missing `discount` ไม่ได้แปลว่า 0 โดยอัตโนมัติ เติม 0 เฉพาะเมื่อ `sales_amount` ตรงกับยอดก่อนส่วนลดภายใน 1 บาท มิฉะนั้นไม่ใช้เป็นยอดขาย
- Quantity ศูนย์/ติดลบไม่ใช่รายการขายที่สมเหตุสมผล จึงตัด
- ราคา outlier เทียบกับ Median ของสินค้าเดียวกัน ไม่ใช้ threshold เดียวทั้งร้าน
- ใช้เฉพาะ `order_status = Completed`; `Cancelled`, `Returned` และ `Unknown` ไม่รวมในยอดขาย
- `net_sales` คำนวณจาก `quantity × unit_price × (1 - discount)` และใช้เป็นตัวเลขหลักหลังตรวจสอบ

## ข้อจำกัด

การตัดรายการที่ไม่สมบูรณ์ทำให้ยอดขายสะท้อนเฉพาะรายการ Completed ที่ผ่านกติกา ข้อมูลจำลองนี้ไม่มีเจ้าของระบบให้ยืนยันเหตุผลของ status หรือส่วนลด จึงต้องตรวจสอบซ้ำเมื่อใช้กับข้อมูลจริง
""")
    write_md(ROOT / "BUSINESS_INSIGHTS.md", f"""# Business Insights และข้อเสนอแนะ — บ้านดีมาร์ท

## Insight 1: ช่องทางหลัก

**Finding:** {top_channel} มียอดขายสูงสุด {money(m['channel_sales'].iloc[0])}  
**Evidence:** ตารางยอดขายตามช่องทางและกราฟ `02_sales_by_channel.png`  
**Interpretation:** ช่องทางนี้เป็นแหล่งรายได้หลักในข้อมูลที่สังเกตได้  
**Implication:** ควรจัดลำดับเวลาและการติดตามช่องทางนี้  
**Recommendation:** วัดยอดขาย คำสั่งซื้อ และ AOV ร่วมกันก่อนเพิ่มทรัพยากร  
**Limitation:** ไม่มีค่าโฆษณา Traffic หรือกำไร จึงสรุป ROI ไม่ได้

## Insight 2: หมวดหมู่สินค้า

**Finding:** {top_category} มียอดขายสูงสุด {money(m['category'].iloc[0])}  
**Evidence:** กราฟ `05_sales_by_category.png`  
**Recommendation:** รักษาสต็อกและทดสอบ Bundle โดยตรวจสอบ Margin จริงก่อน  
**Limitation:** ยอดขายสูงไม่เท่ากับกำไรสูง

## Insight 3: ความแตกต่างรายเดือน

**Finding:** เดือน {best_month} สูงสุด {money(m['month'].loc[best_month])}; เดือน {low_month} ต่ำสุด {money(m['month'].loc[low_month])}  
**Evidence:** กราฟ `01_monthly_sales_trend.png`  
**Recommendation:** ใช้วางแผนสต็อกเบื้องต้นและเก็บข้อมูลหลายปีเพิ่ม  
**Limitation:** ปีเดียวไม่ยืนยันฤดูกาล

## Insight 4: ลูกค้าซื้อซ้ำ

**Finding:** ลูกค้าซื้อซ้ำ {m['repeat']:,} จาก {m['customers']:,} ราย ({pc(m['repeat_rate'])})  
**Evidence:** กราฟ `09_new_vs_repeat_customers.png` และความถี่การซื้อ  
**Recommendation:** ทำ CRM รักษาลูกค้าเดิมและวัดการเปลี่ยนลูกค้าใหม่เป็นลูกค้าซื้อซ้ำ  
**Limitation:** ไม่เห็นประวัติก่อน/หลังช่วงข้อมูล

## Insight 5: การกระจุกตัวของรายได้

**Finding:** สินค้า {m['n80']} จาก {m['products']} รายการสะสมยอดขายถึง 80%  
**Evidence:** ตารางสินค้าและ Pareto ใน Notebook  
**Recommendation:** จัดลำดับการเติมสต็อกสินค้าหลัก แต่ทบทวน Margin ก่อนเลิกสินค้ารายได้ต่ำ  
**Limitation:** ไม่มีต้นทุนสินค้า
""")
    write_md(ROOT / "LEARNING_GUIDE.md", """# Learning Guide — บ้านดีมาร์ท

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
""")


def make_figures(m):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for old in FIG_DIR.glob("*.png"): old.unlink()
    legacy_figures = FIG_DIR / "figures"
    if legacy_figures.exists(): shutil.rmtree(legacy_figures)
    mpl_font = FontProperties(fname=str(FONT_PATH)).get_name() if FONT_PATH.exists() else "DejaVu Sans"
    plt.rcParams["font.family"] = mpl_font
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 11, "axes.unicode_minus": False})
    thai_months = ["ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.", "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค."]
    def save(name): plt.tight_layout(); plt.savefig(FIG_DIR / name, dpi=180, bbox_inches="tight"); plt.close()
    plt.figure(figsize=(10, 5)); plt.plot(range(1, 13), [m["month"].get(i, 0) for i in range(1, 13)], marker="o", color="#007C91"); plt.xticks(range(1, 13), thai_months); plt.title("แนวโน้มยอดขายรายเดือน"); plt.xlabel("เดือน"); plt.ylabel("ยอดขายสุทธิ (บาท)"); plt.grid(axis="y", alpha=.25); save("01_monthly_sales_trend.png")
    for i, (series, title, xlabel, name) in enumerate([(m["channel_sales"], "ยอดขายตามช่องทางการขาย", "ยอดขายสุทธิ (บาท)", "02_sales_by_channel.png"), (m["channel_orders"], "จำนวนคำสั่งซื้อตามช่องทาง", "จำนวนคำสั่งซื้อ", "03_orders_by_channel.png"), (m["channel_aov"], "มูลค่าคำสั่งซื้อเฉลี่ยตามช่องทาง", "AOV (บาท)", "04_aov_by_channel.png"), (m["category"], "ยอดขายตามหมวดหมู่สินค้า", "ยอดขายสุทธิ (บาท)", "05_sales_by_category.png")]):
        plt.figure(figsize=(9, 5)); series.sort_values().plot(kind="barh", color="#E07A5F"); plt.title(title); plt.xlabel(xlabel); plt.ylabel(""); save(name)
    plt.figure(figsize=(10, 6)); m["product"].head(10).sort_values("net_sales").set_index("product_name").net_sales.plot(kind="barh", color="#2A9D8F"); plt.title("สินค้า 10 อันดับแรกตามยอดขาย"); plt.xlabel("ยอดขายสุทธิ (บาท)"); plt.ylabel(""); save("06_top_products.png")
    plt.figure(figsize=(10, 6)); m["region"].sort_values().plot(kind="barh", color="#457B9D"); plt.title("ยอดขายตามภูมิภาค"); plt.xlabel("ยอดขายสุทธิ (บาท)"); plt.ylabel(""); save("07_sales_by_region.png")
    freq = m["customer"].order_count.value_counts().sort_index(); plt.figure(figsize=(9, 5)); freq.plot(kind="bar", color="#F2A65A"); plt.title("ความถี่ในการซื้อของลูกค้า"); plt.xlabel("จำนวนคำสั่งซื้อต่อลูกค้า"); plt.ylabel("จำนวนลูกค้า"); save("08_customer_purchase_frequency.png")
    plt.figure(figsize=(7, 5)); pd.Series({"ลูกค้าใหม่": m["new"], "ลูกค้าที่ซื้อซ้ำ": m["repeat"]}).plot(kind="bar", color=["#8AB17D", "#264653"]); plt.title("ลูกค้าใหม่เทียบกับลูกค้าที่ซื้อซ้ำ"); plt.ylabel("จำนวนลูกค้า"); plt.xticks(rotation=0); save("09_new_vs_repeat_customers.png")
    plt.figure(figsize=(9, 5)); plt.hist(m["orders"].order_value, bins=25, color="#6D597A", alpha=.85); plt.axvline(m["aov"], color="#E07A5F", linestyle="--", label=f"Mean {m['aov']:,.0f}"); plt.axvline(m["median"], color="#007C91", linestyle="--", label=f"Median {m['median']:,.0f}"); plt.title("การกระจายของมูลค่าคำสั่งซื้อ"); plt.xlabel("มูลค่าคำสั่งซื้อ (บาท)"); plt.ylabel("จำนวนคำสั่งซื้อ"); plt.legend(); save("10_order_value_distribution.png")


def make_notebook():
    nb = nbf.v4.new_notebook(); cells = []
    def md(text):
        cell = nbf.v4.new_markdown_cell(text.strip())
        cell.metadata["language"] = "markdown"
        cells.append(cell)

    def code(text):
        cell = nbf.v4.new_code_cell(text.strip())
        cell.metadata["language"] = "python"
        cells.append(cell)
    md("""# บ้านดีมาร์ท — วิเคราะห์ยอดขายและประสิทธิภาพช่องทางการขาย\n\n**Baan Dee Mart — Thai E-commerce Sales Analytics**\n\n> ข้อมูลนี้เป็น **Synthetic Dataset** เพื่อ Portfolio ไม่ใช่ข้อมูลจากบริษัทจริง\n\nNotebook นี้ตอบคำถามธุรกิจด้วย Python/pandas ตั้งแต่ตรวจข้อมูล ทำความสะอาด คำนวณ Metrics วิเคราะห์ลูกค้า สร้างกราฟ และสรุปข้อเสนอแนะ โดยหนึ่งแถวคือสินค้า 1 รายการในคำสั่งซื้อ 1 รายการ""")
    md("""## Phase 1: Business Understanding\n\nบ้านดีมาร์ทเป็นร้านค้าออนไลน์ SME ไทยที่ขายผ่าน Shopee, Lazada, TikTok Shop, Facebook และ Website เจ้าของร้านต้องการรู้ช่องทาง สินค้า พื้นที่ และลูกค้าที่ควรจัดลำดับความสำคัญ\n\n**ข้อจำกัด:** เป็นข้อมูลจำลองปีเดียว ไม่มีต้นทุน กำไร ค่าโฆษณา Traffic และประวัติลูกค้านอกช่วงข้อมูล จึงไม่สรุป Profit หรือ ROI""")
    code("""from pathlib import Path\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\nPROJECT_ROOT = next(\n    (candidate for candidate in [Path.cwd(), *Path.cwd().parents]\n     if (candidate / 'data' / 'raw').is_dir()),\n    Path.cwd(),\n)\nRAW_PATH = PROJECT_ROOT / 'data' / 'raw' / 'orders_raw.csv'\nCLEAN_PATH = PROJECT_ROOT / 'data' / 'cleaned' / 'orders_cleaned.csv'\nFIGURES_DIR = PROJECT_ROOT / 'outputs'\nraw_df = pd.read_csv(RAW_PATH)\nclean_df = pd.read_csv(CLEAN_PATH, parse_dates=['order_date'])\nprint('Project root:', PROJECT_ROOT)\nprint('Raw shape:', raw_df.shape)\nprint('Cleaned shape:', clean_df.shape)""")
    md("""## Phase 2: Data Understanding\n\nตรวจ shape, ตัวอย่างข้อมูล, data types, missing values, duplicate, unique values, ค่าสถิติ และ Grain ก่อนแก้ไขข้อมูล""")
    code("""display(raw_df.head())\ndisplay(raw_df.dtypes.to_frame('data_type'))\ndisplay(raw_df.isna().sum().to_frame('missing_count'))\nprint('Duplicate rows:', raw_df.duplicated().sum())\ndisplay(raw_df[['quantity', 'unit_price', 'discount', 'sales_amount']].describe().T)""")
    md("""## Phase 3: Data Cleaning\n\nใช้เฉพาะ `Completed` เป็นยอดขาย ตัดวันที่อ่านไม่ได้ duplicate quantity ที่ไม่ถูกต้อง และราคา outlier ตรวจ `sales_amount` กับสูตร `quantity × unit_price × (1 - discount)` โดยไม่เติม missing discount เป็นศูนย์โดยอัตโนมัติ""")
    code("""print('Missing after cleaning:', clean_df.isna().sum().sum())\nprint('Duplicates after cleaning:', clean_df.duplicated().sum())\nprint(clean_df.order_status.value_counts(dropna=False))""")
    md("""## Phase 4-5: Feature Engineering และ Sales Analysis\n\nคำนวณ `gross_sales`, `net_sales`, เดือน, customer type และรวม Metrics ที่ระดับ Order เพื่อไม่ให้นับ Order Line เป็น Order""")
    code("""orders = clean_df.groupby('order_id', as_index=False).agg(order_date=('order_date','first'), customer_id=('customer_id','first'), channel=('channel','first'), order_value=('net_sales','sum'))\nmetrics = {\n    'ยอดขายสุทธิ': clean_df.net_sales.sum(),\n    'คำสั่งซื้อ': orders.order_id.nunique(),\n    'ลูกค้า': clean_df.customer_id.nunique(),\n    'AOV': orders.order_value.mean(),\n    'Median Order Value': orders.order_value.median(),\n}\npd.Series(metrics)""")
    code("""sales_by_channel = clean_df.groupby('channel').net_sales.sum().sort_values(ascending=False)\norders_by_channel = orders.groupby('channel').order_id.nunique().sort_values(ascending=False)\naov_by_channel = orders.groupby('channel').order_value.mean().sort_values(ascending=False)\ndisplay(sales_by_channel.to_frame('net_sales'))\ndisplay(orders_by_channel.to_frame('order_count'))\ndisplay(aov_by_channel.to_frame('aov'))""")
    code("""sales_by_category = clean_df.groupby('category').net_sales.sum().sort_values(ascending=False)\nsales_by_product = clean_df.groupby(['product_id', 'product_name']).net_sales.sum().sort_values(ascending=False)\nsales_by_province = clean_df.groupby('province').net_sales.sum().sort_values(ascending=False)\nsales_by_region = clean_df.groupby('region').net_sales.sum().sort_values(ascending=False)\nmonthly_sales = clean_df.groupby('order_month').net_sales.sum()\ndisplay(sales_by_category.to_frame('net_sales'))\ndisplay(sales_by_product.head(10).to_frame('net_sales'))\ndisplay(sales_by_province.head(10).to_frame('net_sales'))\ndisplay(sales_by_region.to_frame('net_sales'))\ndisplay(monthly_sales.to_frame('net_sales'))""")
    md("""## Phase 6: Customer Analysis\n\nลูกค้าใหม่หมายถึงลูกค้าที่มี 1 Order ในช่วงข้อมูล และลูกค้าที่ซื้อซ้ำมีมากกว่า 1 Order ไม่ใช่สถานะตลอดอายุลูกค้า""")
    code("""customer_summary = orders.groupby('customer_id').agg(order_count=('order_id','nunique'), monetary=('order_value','sum'))\ncustomer_summary['customer_type'] = np.where(customer_summary.order_count > 1, 'ลูกค้าที่ซื้อซ้ำ', 'ลูกค้าใหม่')\ndisplay(customer_summary.groupby('customer_type').size().to_frame('customer_count'))""")
    md("""## Phase 7-8: Statistical Thinking และ Visualization\n\nMean สูงกว่า Median อาจสะท้อนคำสั่งซื้อขนาดใหญ่บางรายการ และ Correlation ไม่ได้แปลว่า Causation กราฟภาษาไทย 10 รายการถูกบันทึกไว้ใน `outputs/`""")
    code("""print('Mean:', orders.order_value.mean())\nprint('Median:', orders.order_value.median())\nprint('Std:', orders.order_value.std())\nprint('75th percentile:', orders.order_value.quantile(.75))\nfrom IPython.display import display, Image\nfigure_names = [\n    '01_monthly_sales_trend.png', '02_sales_by_channel.png', '03_orders_by_channel.png',\n    '04_aov_by_channel.png', '05_sales_by_category.png', '06_top_products.png',\n    '07_sales_by_region.png', '08_customer_purchase_frequency.png',\n    '09_new_vs_repeat_customers.png', '10_order_value_distribution.png'\n]\nfor figure_name in figure_names:\n    display(Image(filename=str(FIGURES_DIR / figure_name), width=700))""")
    md("""## Phase 9-10: Business Insights และ Executive Summary\n\nผลสรุปใน Notebook ต้องอ่านคู่กับ `BUSINESS_INSIGHTS.md`, `EXECUTIVE_SUMMARY.md` และ PDF โดยทุกตัวเลขดึงจาก Cleaned Dataset และทุกข้อเสนอแนะระบุข้อจำกัดไว้ชัดเจน""")
    nb["cells"] = cells; nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}}
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True); nbf.write(nb, NOTEBOOK)


def make_pdfs(m, audit):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pdfmetrics.registerFont(TTFont("ThaiUI", str(FONT_PATH)))
    pdfmetrics.registerFont(TTFont("ThaiUI-Bold", str(FONT_BOLD_PATH)))
    styles = {
        "title": ParagraphStyle("title", fontName="ThaiUI-Bold", fontSize=20, leading=26, textColor=colors.HexColor("#174A5B"), spaceAfter=9),
        "h": ParagraphStyle("h", fontName="ThaiUI-Bold", fontSize=13, leading=18, textColor=colors.HexColor("#174A5B"), spaceBefore=10, spaceAfter=5),
        "body": ParagraphStyle("body", fontName="ThaiUI", fontSize=9.5, leading=14, textColor=colors.HexColor("#263238"), spaceAfter=6),
        "small": ParagraphStyle("small", fontName="ThaiUI", fontSize=8, leading=11, textColor=colors.HexColor("#52606D")),
        "table": ParagraphStyle("table", fontName="ThaiUI", fontSize=8.7, leading=11, textColor=colors.HexColor("#263238")),
        "table_head": ParagraphStyle("table_head", fontName="ThaiUI-Bold", fontSize=8.7, leading=11, textColor=colors.white),
    }

    def p(text, style="body"):
        safe_text = str(text).replace("&", "&amp;").replace("<br/>", "@@BR@@").replace("<", "&lt;").replace(">", "&gt;").replace("@@BR@@", "<br/>")
        return Paragraph(safe_text, styles[style])

    def report_table(rows, widths):
        data = [[p(value, "table_head") for value in rows[0]]]
        data += [[p(value, "table") for value in row] for row in rows[1:]]
        table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#174A5B")),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CAD5DB")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        return table

    def chart(story, filename, caption):
        story.extend([Image(str(FIG_DIR / filename), width=15.2 * cm, height=7.45 * cm), p("กราฟ: " + caption, "small"), Spacer(1, 8)])

    top_channel = m["channel_sales"].index[0]
    top_category = m["category"].index[0]
    best_month = m["month"].idxmax()
    low_month = m["month"].idxmin()
    common = [p("Synthetic Dataset | บ้านดีมาร์ท", "small")]
    reports = {
        "บ้านดีมาร์ท_รายงานวิเคราะห์ธุรกิจ.pdf": [
            p("บ้านดีมาร์ท — รายงานวิเคราะห์ธุรกิจ", "title"), p("Baan Dee Mart — Thai E-commerce Sales Analytics | มกราคม-ธันวาคม 2024", "body"), *common,
            p("Project Overview", "h"), p("รายงานนี้วิเคราะห์ร้านค้าออนไลน์ SME ไทยจากข้อมูลจำลองระดับ Order Line เพื่อสนับสนุนการวางแผนช่องทาง สินค้า พื้นที่ และลูกค้า", "body"),
            p("Business Problem และ Objective", "h"), p("เจ้าของร้านต้องการทราบว่าช่องทางใดสร้างยอดขายสูง สินค้ากลุ่มใดควรรักษา ตลาดใดสำคัญ และลูกค้าซื้อซ้ำมากน้อยเพียงใด โดยต้องระวังการนับหลายแถวต่อหนึ่ง Order", "body"),
            p("Key Metrics", "h"), report_table([["ตัวชี้วัด", "ผลลัพธ์"], ["ยอดขายสุทธิ", money(m["sales"])], ["คำสั่งซื้อ", f"{m['order_count']:,}"], ["ลูกค้า", f"{m['customers']:,}"], ["AOV / Median", f"{money(m['aov'])} / {money(m['median'])}"], ["Repeat Rate", pc(m["repeat_rate"])]], [8 * cm, 8 * cm]),
            p("Key Findings", "h"), p(f"• {top_channel} เป็นช่องทางยอดขายสูงสุด ({money(m['channel_sales'].iloc[0])})<br/>• {top_category} เป็นหมวดหมู่ยอดขายสูงสุด ({money(m['category'].iloc[0])})<br/>• เดือน {best_month} สูงสุด และเดือน {low_month} ต่ำสุด<br/>• ลูกค้าซื้อซ้ำคิดเป็น {pc(m['repeat_rate'])}<br/>• จังหวัดยอดขายสูงสุดคือ {m['province'].index[0]}", "body"),
            p("ข้อสรุป", "h"), p("ข้อมูลสนับสนุนการจัดลำดับช่องทาง สินค้า และการรักษาลูกค้าในเชิง Descriptive Analytics แต่ควรเติมข้อมูลต้นทุนและหลายช่วงเวลา ก่อนใช้เป็นแผนลงทุนระยะยาว", "body"), PageBreak(), p("กราฟและหลักฐาน", "h")
        ],
        "บ้านดีมาร์ท_บทสรุปผู้บริหาร.pdf": [
            p("บ้านดีมาร์ท — บทสรุปผู้บริหาร", "title"), p("รายงานสำหรับเจ้าของร้านและผู้ว่าจ้าง | ข้อมูลจำลองปี 2024", "body"), *common,
            p("ภาพรวม", "h"), p(f"วิเคราะห์ {m['order_count']:,} คำสั่งซื้อจากลูกค้า {m['customers']:,} ราย ยอดขายสุทธิรวม {money(m['sales'])} โดยใช้เฉพาะสถานะ Completed", "body"),
            p("ผลลัพธ์สำคัญ", "h"), p(f"• ช่องทางหลัก: {top_channel}<br/>• หมวดหมู่หลัก: {top_category}<br/>• ยอดขายสูงสุด: เดือน {best_month} ({money(m['month'].loc[best_month])})<br/>• Repeat Purchase Rate: {pc(m['repeat_rate'])}<br/>• AOV: {money(m['aov'])} และ Median: {money(m['median'])}", "body"),
            p("ข้อเสนอแนะ", "h"), p(f"1. จัดลำดับการดูแล {top_channel} โดยวัดยอดขาย คำสั่งซื้อ และ AOV ร่วมกัน<br/>2. รักษาสต็อก {top_category} และทดสอบ Bundle โดยตรวจ Margin<br/>3. วางแผนรายเดือนและเก็บข้อมูลเพิ่มเพื่อยืนยันฤดูกาล<br/>4. ทำ CRM ลูกค้าซื้อซ้ำและวัดผลลูกค้าใหม่", "body"),
            p("ข้อจำกัด", "h"), p("เป็น Synthetic Dataset ปีเดียว ไม่มีต้นทุน กำไร ค่าโฆษณา Traffic และประวัติลูกค้านอกช่วงข้อมูล จึงไม่ใช่รายงาน Profit หรือ ROI", "body"), PageBreak(), p("กราฟประกอบ", "h")
        ],
        "บ้านดีมาร์ท_รายงานการทำความสะอาดข้อมูล.pdf": [
            p("บ้านดีมาร์ท — รายงานการทำความสะอาดข้อมูล", "title"), p("Traceable Data Preparation | Synthetic Dataset", "body"),
            p("ผลลัพธ์", "h"), report_table([["รายการ", "จำนวน"], ["Raw rows", f"{audit['raw_rows']:,}"], ["Cleaned rows", f"{audit['clean_rows']:,}"], ["วันที่อ่านไม่ได้", str(audit['unparseable_dates'])], ["Duplicate", str(audit['duplicates'])], ["Missing discount ที่พบ", str(audit['missing_discount'])], ["Missing discount ที่เติมได้", str(audit['discount_filled'])], ["Missing discount ที่ยืนยันไม่ได้", str(audit['discount_unresolved'])], ["Invalid quantity", str(audit['invalid_quantity'])], ["Price outlier", str(audit['price_outliers'])], ["ไม่ใช่ Completed", str(audit['cancelled_or_returned'])], ["Sales mismatch", str(audit['sales_mismatches'])]], [8 * cm, 8 * cm]),
            p("กติกาสำคัญ", "h"), p("ใช้เฉพาะ Completed, ไม่เติม Missing discount เป็น 0 โดยอัตโนมัติ, ตรวจราคาเทียบ Median สินค้าเดียวกัน, ตัด Quantity ศูนย์/ติดลบ และคำนวณ net_sales จากฟิลด์ต้นทางหลังตรวจสอบความสอดคล้อง", "body"),
            p("ข้อจำกัด", "h"), p("ข้อมูลจำลองไม่มีเจ้าของระบบให้ยืนยันเหตุผลของ Status หรือ Discount จึงต้องตรวจสอบซ้ำก่อนใช้ในงานจริง", "body"), PageBreak(), p("กราฟตรวจสอบประกอบ", "h")
        ],
    }
    for filename, story in reports.items():
        if "รายงานวิเคราะห์" in filename:
            chart(story, "01_monthly_sales_trend.png", "แนวโน้มยอดขายรายเดือน")
            story.append(PageBreak())
            chart(story, "02_sales_by_channel.png", "ยอดขายตามช่องทาง")
        elif "บทสรุป" in filename:
            chart(story, "05_sales_by_category.png", "ยอดขายตามหมวดหมู่")
        else:
            chart(story, "10_order_value_distribution.png", "การกระจายมูลค่าคำสั่งซื้อ")
        story += [p("Business Recommendations และ Limitations", "h"), p("คำแนะนำทั้งหมดเป็นข้อเสนอแนะจาก Descriptive Analytics ไม่ใช่การยืนยันเหตุและผล ควรเติมข้อมูลต้นทุน Margin ค่าโฆษณา Traffic และข้อมูลหลายช่วงเวลาเพื่อยกระดับการตัดสินใจ", "body")]
        doc = SimpleDocTemplate(str(REPORT_DIR / filename), pagesize=A4, rightMargin=1.7 * cm, leftMargin=1.7 * cm, topMargin=1.4 * cm, bottomMargin=1.5 * cm, title=filename)

        def footer(canvas, doc):
            canvas.saveState()
            canvas.setFont("ThaiUI", 7)
            canvas.setFillColor(colors.HexColor("#52606D"))
            canvas.drawString(1.7 * cm, .8 * cm, "บ้านดีมาร์ท | Synthetic Dataset")
            canvas.drawRightString(A4[0] - 1.7 * cm, .8 * cm, f"หน้า {doc.page}")
            canvas.restoreState()

        doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    for old_path in [
        ROOT / "data" / "raw" / "brightcart_orders_raw.csv",
        ROOT / "data" / "cleaned" / "brightcart_orders_cleaned.csv",
        ROOT / "notebooks" / "brightcart_analysis.ipynb",
    ]:
        if old_path.exists():
            old_path.unlink()
    raw = make_raw(); clean, audit = clean_data(raw); m = metrics(clean); make_figures(m); build_docs(m, audit); make_notebook(); make_pdfs(m, audit)
    print(f"สร้าง Raw {len(raw):,} แถว -> Cleaned {len(clean):,} แถว")
    print(f"ยอดขายสุทธิ {money(m['sales'])}, Orders {m['order_count']:,}, Customers {m['customers']:,}, Repeat {pc(m['repeat_rate'])}")


if __name__ == "__main__": main()
