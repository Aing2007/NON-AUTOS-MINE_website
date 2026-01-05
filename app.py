# rag_chat.py
from __future__ import annotations
import os
from pathlib import Path
from typing import List, Tuple

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# retriever ของคุณ: vector.md_rag(source_directory, db_path, collection_name) -> retriever (มี .invoke(query))
from vector import md_rag


# =========================
# CONFIG (แก้ได้ตามสะดวก หรือใช้ ENV)
# =========================
OLLAMA_API_BASE = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
OLLAMA_MODEL    = os.environ.get("OLLAMA_MODEL", "llama3.2")

MD_FILES = os.environ.get(
    "MD_FILES",
    "/Users/sutinan/Desktop/Project/NON-AUTOS-MINE(Overall)/Backend/RAGLLM/markdown_file",
)
DB_PATH       = os.environ.get("CHROMA_DB_PATH", "chroma_md_db")
COLLECTION    = os.environ.get("CHROMA_COLLECTION", "tech_files")
MAX_SNIPPET   = int(os.environ.get("MAX_SNIPPET", "1200"))
TOP_K         = int(os.environ.get("TOP_K", "4"))  # จำนวนเอกสารที่ใช้เป็นบริบท


# =========================
# PROMPT
# =========================
TEMPLATE = """
ระบบ: คุณคือผู้ช่วยของโครงการ NON-AUTOS-MINE ด้านออทิสติก/พัฒนาการเด็ก
ตอบโดยอิงข้อมูลจากเอกสาร Markdown ที่ให้ "เท่านั้น" ห้ามเดาจากความจำหรือเว็บภายนอก

บริบทเอกสาร (Markdown):
{documents}

คำถามของผู้ใช้:
{question}

ข้อกำหนดการตอบ (สำคัญ):
- ภาษาไทย สุภาพ ชัดเจน เหมาะกับครู/ผู้ปกครอง/นักการศึกษา
- ถ้าไม่พบคำตอบในบริบท ให้ตอบว่า: "ไม่พบคำตอบในบริบทที่ให้มา"
  แล้วเสนอคีย์เวิร์ด/หัวข้อที่ควรค้นเพิ่มภายในคลังเอกสารของเรา
- โครงสร้างคำตอบ (Markdown):
  1) สรุปสั้น (2–4 บรรทัด)
  2) รายละเอียด/ขั้นตอน/ข้อควรระวัง (รายการย่อยได้)
  3) อ้างอิงไฟล์ที่ใช้ (ระบุชื่อไฟล์ Markdown)
- การอ้างอิง: วาง [filename.md] ต่อท้ายประโยค/ย่อหน้าที่อ้างข้อเท็จจริง
  (ถ้าใช้หลายไฟล์ ให้ใส่ทุกไฟล์ที่เกี่ยวข้อง เช่น [guide_autism.md][screening_th.md])
- ถ้าคำถามเป็นการเปรียบเทียบ/สรุปรายการ ให้จัดเป็นตารางเมื่อเหมาะสม
- ระบุตัวเลขพร้อมหน่วย/ช่วงเวลาให้ชัด
- หลีกเลี่ยงถ้อยคำเหมือนการวินิจฉัยโรคทางการแพทย์ ให้เป็นข้อมูลทั่วไป/แนวทางเบื้องต้น
- ห้ามเปิดเผยขั้นตอนการคิดภายใน ให้แสดงเฉพาะคำตอบปลายทางอย่างมีที่มา

รูปแบบเอาต์พุต:
- ใช้ Markdown, ไม่ใส่โค้ดเทียม/เมตา
- ความยาวโดยค่าเริ่มต้น 150–300 คำ (ขยายได้ถ้าคำถามระบุว่าต้องการรายละเอียดมาก)

ตอนนี้จงตอบ:
"""
prompt = ChatPromptTemplate.from_template(TEMPLATE)


# =========================
# HELPERS
# =========================
def build_context(docs, max_chars: int = MAX_SNIPPET, top_k: int = TOP_K) -> Tuple[str, List[str]]:
    """รวมข้อความเอกสารเป็น context + คืนรายชื่อไฟล์ที่ใช้"""
    if not docs:
        return "(no relevant context found)", []
    docs = docs[: top_k]
    pieces = []
    sources = []
    for d in docs:
        src = Path(d.metadata.get("source", "")).name or "unknown.md"
        text = d.page_content or ""
        snippet = text[:max_chars] + ("..." if len(text) > max_chars else "")
        pieces.append(f"[source: {src}]\n{snippet}")
        sources.append(src)
    context = "\n\n---\n\n".join(pieces)
    # เรียงและลบซ้ำ
    sources = sorted(list(dict.fromkeys(sources)))
    return context, sources


def make_score_question(score: float) -> str:
    return (
        f"คะแนนจากกิจกรรมพัฒนาการคือ {score} "
        f"ซึ่งเป็นการวัดสมาธิของเด็กขณะที่ทำกิจกรรม ให้ทำการวิเคราะห์โดยแบ่งตามเกณฑ์แบบทดสอบ ATEC "
        f"เป็นหลัก พร้อมชี้แจงเหตุผล"
    )


# =========================
# MAIN
# =========================
def main():
    # 1) ตรวจโฟลเดอร์เอกสาร
    md_dir = Path(MD_FILES).resolve()
    if not md_dir.exists():
        print(f"[ERROR] Source directory not found: {md_dir}")
        return

    # 2) โหลดโมเดล + ตัวดึงข้อมูล
    print(f"[INFO] Using Ollama model: {OLLAMA_MODEL} @ {OLLAMA_API_BASE}")
    print(f"[INFO] Markdown dir: {md_dir}")
    print(f"[INFO] Building retriever (db='{DB_PATH}', collection='{COLLECTION}') ...")
    try:
        retriever = md_rag(
            source_directory=str(md_dir),
            db_path=DB_PATH,
            collection_name=COLLECTION,
        )
        print("[INFO] Retriever ready.")
    except Exception as e:
        print(f"[ERROR] Failed to build retriever: {e}")
        return

    model = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_API_BASE)
    chain = prompt | model

    # 3) REPL
    print("\n================= RAG Chat (Terminal) =================")
    print("พิมพ์คำถามแล้วกด Enter เพื่อถาม")
    print("คำสั่งพิเศษ:")
    print("  /score <ตัวเลข>    → สร้างคำถามวิเคราะห์คะแนนตาม ATEC")
    print("  /k <จำนวน>         → ปรับจำนวนเอกสารบริบท (TOP_K) ขณะรัน")
    print("  /exit หรือ q        → ออกจากโปรแกรม")
    print("=======================================================\n")

    top_k = TOP_K
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Bye]")
            break

        if not question:
            continue
        if question.lower() in {"q", "/exit", "exit"}:
            print("[Bye]")
            break

        # คำสั่งพิเศษ
        if question.startswith("/score"):
            parts = question.split()
            if len(parts) >= 2:
                try:
                    sc = float(parts[1])
                    question = make_score_question(sc)
                    print(f"[CMD] ใช้คำถามแบบ ATEC ด้วยคะแนน = {sc}")
                except ValueError:
                    print("[WARN] รูปแบบ /score ไม่ถูกต้อง เช่น /score 78")
                    continue
            else:
                print("[WARN] ใช้รูปแบบ: /score 78")
                continue
        elif question.startswith("/k"):
            parts = question.split()
            if len(parts) >= 2 and parts[1].isdigit():
                top_k = max(1, int(parts[1]))
                print(f"[CMD] ปรับ TOP_K = {top_k}")
            else:
                print("[WARN] ใช้รูปแบบ: /k 4")
            continue

        # 4) ดึงเอกสาร + สร้าง context
        try:
            docs = retriever.invoke(question)
        except Exception as e:
            print(f"[ERROR] retriever failed: {e}")
            continue

        context, sources = build_context(docs, MAX_SNIPPET, top_k)

        # 5) เรียก LLM ตอบ
        try:
            answer = chain.invoke({"documents": context, "question": question})
            if not isinstance(answer, str):
                answer = str(answer)
        except Exception as e:
            print(f"[ERROR] LLM failed: {e}")
            continue

        # 6) แสดงผล
        print("\n--- ANSWER (Markdown) ---")
        print(answer.strip())
        if sources:
            print("\n[Sources]", ", ".join(sources))
        print("-------------------------\n")


if __name__ == "__main__":
    main()
