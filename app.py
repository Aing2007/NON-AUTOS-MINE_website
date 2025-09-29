from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

from vector import md_rag
from pathlib import Path
import os, sys, traceback

ollama_url = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
model = OllamaLLM(model="llama3.2", base_url=ollama_url)

template = """
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

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

md_files = "/Users/sutinan/Desktop/Project/NON-AUTOS-MINE(Overall)/RAGLLM/markdown_file"  # ← path ในเครื่องคุณเอง :contentReference[oaicite:7]{index=7}
print("md_files path =", md_files)

p = Path(md_files).resolve()
if not p.exists():
    print(f"[ERROR] Source directory not found: {p}")
    sys.exit(1)

try:
    retriever = md_rag(
        source_directory=md_files,
        db_path="chroma_md_db",
        collection_name="tech_files"
    )
    print("Retriever ready.")
except Exception as e:
    print("[ERROR] Failed to build retriever:", e)
    traceback.print_exc()
    sys.exit(1)

while True:
    print("\n-------------------------------")
    question = input("Ask your question (q to quit): ").strip()
    if question.lower() == "q":
        break

    # ดึงเอกสารที่เกี่ยวข้อง
    docs = retriever.invoke(question)   # list[Document]
    # แปลงเป็นสตริง (ตัดทอน + ใส่ metadata ชื่อไฟล์)
    context = "\n\n---\n\n".join(
        f"[source: {Path(d.metadata.get('source','')).name}]\n{d.page_content[:1200]}"
        + ("..." if len(d.page_content) > 1200 else "")
        for d in docs
    ) or "(no relevant context found)"

    result = chain.invoke({"documents": context, "question": question})
    print(f"AI: {result}")
