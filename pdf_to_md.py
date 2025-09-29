#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch convert PDFs in a folder -> Markdown (.md) with Thai support.
- ใช้ PyMuPDF ดึงข้อความ (โหมด "markdown") เพื่อเก็บโครงสร้างหัวข้อ/ลิสต์เท่าที่เป็นไปได้
- ถ้าดึงไม่ได้ (เช่นไฟล์สแกน) จะ fallback เป็น OCR (pytesseract, lang=tha+eng)
- เขียน front matter ด้านบนไฟล์ .md เพื่อเก็บ metadata เบื้องต้น
"""

import os, sys, re, argparse, datetime
from pathlib import Path
from typing import Optional
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from tqdm import tqdm

def safe_stem(name: str) -> str:
    # แปลงชื่อไฟล์ให้ปลอดภัยต่อไฟล์ระบบ
    stem = Path(name).stem
    stem = re.sub(r"[^\wก-๙\-]+", "_", stem, flags=re.UNICODE)
    return stem.strip("_") or "document"

def extract_markdown_from_pdf(pdf_path: Path, ocr: bool, ocr_lang: str = "tha+eng") -> str:
    """
    พยายามดึงเป็น markdown ด้วย PyMuPDF ก่อน
    ถ้าน้อย/ว่าง -> ทำ OCR ต่อหน้า (render เป็นภาพแล้ว OCR)
    """
    doc = fitz.open(pdf_path)
    out_lines = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]

        # 1) ลองดึงเป็น markdown โดยตรง (รักษา bullet/heading เท่าที่ได้)
        md = page.get_text("markdown") or ""
        # เคสบางไฟล์ md จะว่าง/น้อยมาก ลองใช้ text แบบ block แทน
        if len(md.strip()) < 10:
            md = page.get_text("text") or ""

        # ถ้าดึงออกมาได้น้อยมาก และอนุญาต OCR ให้ทำ OCR ต่อหน้า
        needs_ocr = (len(md.strip()) < 10) and ocr

        if not needs_ocr:
            # แทรกหัวข้อหน้า
            out_lines.append(f"\n\n## หน้า {page_idx+1}\n")
            out_lines.append(md.strip())
        else:
            # 2) OCR: render หน้าเป็นภาพแล้วให้ Tesseract อ่าน
            # Render ที่ DPI สูงหน่อยเพื่อช่วย OCR
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr_text = pytesseract.image_to_string(img, lang=ocr_lang)
            out_lines.append(f"\n\n## หน้า {page_idx+1} (OCR)\n")
            out_lines.append(ocr_text.strip())

    doc.close()
    return "\n".join(out_lines).strip()

def write_markdown(out_path: Path, content: str, source_pdf: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    created = datetime.datetime.now().isoformat(timespec="seconds")
    front_matter = (
        "---\n"
        f"title: \"{source_pdf.stem}\"\n"
        f"source_pdf: \"{str(source_pdf)}\"\n"
        f"generated_at: \"{created}\"\n"
        "language: \"th\"\n"
        "---\n\n"
    )
    with out_path.open("w", encoding="utf-8") as f:
        f.write(front_matter + content + "\n")

def is_pdf(p: Path) -> bool:
    return p.suffix.lower() == ".pdf"

def main():
    ap = argparse.ArgumentParser(description="Convert PDFs in a folder to Markdown (.md) with Thai support")
    ap.add_argument("--in_dir", required=True, help="โฟลเดอร์ที่มีไฟล์ .pdf")
    ap.add_argument("--out_dir", required=True, help="โฟลเดอร์ปลายทางสำหรับ .md")
    ap.add_argument("--ocr", action="store_true", help="เปิด OCR อัตโนมัติเมื่อดึงข้อความไม่ออก (แนะนำ)")
    ap.add_argument("--no-ocr", dest="ocr", action="store_false", help="ปิด OCR เพื่อความเร็ว")
    ap.set_defaults(ocr=True)
    ap.add_argument("--ocr_lang", default="tha+eng", help="ภาษา OCR (ค่าเริ่มต้น: tha+eng)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not in_dir.exists():
        print(f"ไม่พบโฟลเดอร์: {in_dir}", file=sys.stderr)
        sys.exit(1)

    pdf_files = sorted([p for p in in_dir.rglob("*.pdf") if p.is_file()])
    if not pdf_files:
        print("ไม่พบไฟล์ .pdf ในโฟลเดอร์ที่ระบุ", file=sys.stderr)
        sys.exit(1)

    print(f"พบ PDF {len(pdf_files)} ไฟล์  →  แปลงเป็น Markdown ไปที่: {out_dir}")
    for pdf_path in tqdm(pdf_files, unit="file"):
        try:
            md_text = extract_markdown_from_pdf(pdf_path, ocr=args.ocr, ocr_lang=args.ocr_lang)
            if not md_text.strip():
                md_text = "_(ไม่พบข้อความในไฟล์นี้)_"

            out_name = f"{safe_stem(pdf_path.name)}.md"
            out_path = out_dir / out_name
            write_markdown(out_path, md_text, pdf_path)
        except Exception as e:
            # ถ้าไฟล์ใดพัง จะไม่หยุดทั้ง batch
            err_name = f"{safe_stem(pdf_path.name)}.error.txt"
            (out_dir / err_name).parent.mkdir(parents=True, exist_ok=True)
            with (out_dir / err_name).open("w", encoding="utf-8") as ef:
                ef.write(f"Error: {e}\n")
            continue

    print("เสร็จสิ้น ✅")

if __name__ == "__main__":
    main()
