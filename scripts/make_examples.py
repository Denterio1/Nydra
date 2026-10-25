import csv, io, random, zipfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

random.seed(42)
out = Path("examples")
out.mkdir(exist_ok=True)

# 1. Sales CSV: small, with missing values and one duplicate row
rows = [["order_id", "region", "units", "price"]]
for i in range(1, 41):
    rows.append([i, random.choice(["north", "south", "east", "west"]),
                 random.randint(1, 20) if i % 9 else "",
                 round(random.uniform(5, 50), 2)])
rows.append(rows[5])
with open(out / "demo_sales.csv", "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(rows)

# 2. Churn CSV: ML-style table with missing values, outliers, duplicates
rows = [["customer_id", "age", "tenure_months", "monthly_charges",
         "contract", "support_calls", "churned"]]
for i in range(1, 301):
    tenure = random.randint(1, 72)
    contract = random.choice(["monthly", "yearly", "two_year"])
    charges = round(random.uniform(20, 110), 2)
    if i % 97 == 0:
        charges = round(charges * 12, 2)  # outlier
    p = 0.45 if contract == "monthly" else 0.12
    p += 0.2 if tenure < 12 else 0
    rows.append([i,
                 random.randint(18, 80) if random.random() > 0.08 else "",
                 tenure,
                 charges if random.random() > 0.06 else "",
                 contract, random.randint(0, 8),
                 int(random.random() < p)])
rows.extend(rows[10:14])
with open(out / "demo_churn.csv", "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(rows)

# 3. Messy text: noise, HTML, duplicates, fake contact details
words = ("data quality model training report customer order delivery "
         "service price market review team project result analysis "
         "system update issue support").split()
lines = []
for i in range(200):
    s = " ".join(random.choices(words, k=random.randint(8, 16))).capitalize() + "."
    if i % 25 == 0:
        s = "<p>" + s + "</p>"
    if i % 40 == 0:
        s += " Contact: user%d@example.com or 555-01%02d." % (i, i % 100)
    lines.append(s)
lines.extend(lines[:15])
(out / "demo_messy_text.txt").write_text("\n".join(lines), encoding="utf-8")

# 4. Images: two classes, plus a dark, a blurry, a duplicate and a corrupt file
def make(kind):
    img = Image.new("RGB", (224, 224), tuple(random.randint(150, 240) for _ in range(3)))
    d = ImageDraw.Draw(img)
    for _ in range(random.randint(2, 4)):
        x, y = random.randint(0, 150), random.randint(0, 150)
        c = tuple(random.randint(0, 120) for _ in range(3))
        if kind == "circles":
            d.ellipse([x, y, x + 70, y + 70], fill=c)
        else:
            d.rectangle([x, y, x + 20, y + 90], fill=c)
    return img

zpath = out / "demo_images.zip"
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for kind in ("circles", "stripes"):
        for i in range(12):
            img = make(kind)
            if kind == "circles" and i == 0:
                img = ImageEnhance.Brightness(img).enhance(0.15)
            if kind == "stripes" and i == 0:
                img = img.filter(ImageFilter.GaussianBlur(8))
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=90)
            z.writestr("demo_images/%s/%s_%02d.jpg" % (kind, kind, i), buf.getvalue())
            if kind == "circles" and i == 1:
                z.writestr("demo_images/%s/%s_01_copy.jpg" % (kind, kind), buf.getvalue())
    z.writestr("demo_images/stripes/broken.jpg", b"this is not an image")

print("OK")