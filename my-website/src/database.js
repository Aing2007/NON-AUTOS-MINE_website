// database.js
// รองรับการบันทึกคะแนน game1–game4 ลงในเอกสารเดียวต่อ (userId × animalId)

import { initializeApp } from "https://www.gstatic.com/firebasejs/10.14.1/firebase-app.js";
import {
    getFirestore,
    doc,
    getDoc,
    setDoc,
    updateDoc,
    serverTimestamp,
} from "https://www.gstatic.com/firebasejs/10.14.1/firebase-firestore.js";

/* ========================= Firebase Config ========================= */
const firebaseConfig = {
    apiKey: "AIzaSyAafwo9zrpeiTu11qpafsyOe6lN1yZsexU",
    authDomain: "nonwebsite-5077e.firebaseapp.com",
    databaseURL: "https://nonwebsite-5077e-default-rtdb.asia-southeast1.firebasedatabase.app",
    projectId: "nonwebsite-5077e",
    storageBucket: "nonwebsite-5077e.appspot.com",
    messagingSenderId: "853304554751",
    appId: "1:853304554751:web:a314f134e9287ae80d48ca",
    measurementId: "G-S7T20G4D3J",
};

const app = initializeApp(firebaseConfig);
const db = getFirestore(app);

/* ========================= Helpers ========================= */
// ทำคะแนนให้อยู่ในช่วง 0–100 (จำนวนเต็ม)
const normScore = (n) =>
    typeof n === "number" ? Math.max(0, Math.min(100, Math.round(n))) : 0;

// ทำให้ userId/animalId คงรูปแบบเสมอ เพื่อลดปัญหาซ้ำซ้อน
const normIdStr = (s, fallback) =>
    String(s ?? fallback)
        .trim()
        .toLowerCase()
        .replace(/\s+/g, "_")
        .replace(/[^\w\-\.@]/g, ""); // เก็บเฉพาะอักขระที่ปลอดภัย

const keyOf = (userId, animalId) =>
    `${normIdStr(userId, "anonymous")}__${normIdStr(animalId, "unknown")}`;

// คอลเลกชันปลายทาง (ใช้ชื่อเดิมตามที่คุณเคยใช้)
const COLLECTION = "game1_results";

/**
 * upsertScore
 * สร้าง/อัปเดตเอกสารผลคะแนนรายสัตว์ของผู้ใช้เดียว (doc id คงที่ = userId__animalId)
 * @param {Object} p
 * @param {string} p.userId
 * @param {string} p.animalId
 * @param {number} p.score
 * @param {'game1'|'game2'|'game3'|'game4'} p.gameKey  ฟิลด์คะแนนที่ต้องการอัปเดต
 * @param {Object} [p.extra]  ฟิลด์เสริม (เช่น source/page/animalName/animalImage)
 */
async function upsertScore({ userId, animalId, score, gameKey, extra = {} }) {
    if (!["game1", "game2", "game3", "game4"].includes(gameKey)) {
        throw new Error(`Invalid gameKey: ${gameKey}`);
    }

    const uid = normIdStr(userId, "anonymous");
    const aid = normIdStr(animalId, "unknown");
    const docId = keyOf(uid, aid);
    const ref = doc(db, COLLECTION, docId);

    const s = normScore(score);
    const now = serverTimestamp();
    const base = {
        userId: uid,
        Animal_ID: aid,
        updatedAt: now,
        ...extra, // สามารถส่ง { source, page, ts, animalName, animalImage } มาเก็บร่วมได้
    };

    const snap = await getDoc(ref);

    if (snap.exists()) {
        // เคยมีเอกสารนี้แล้ว → อัปเดตเฉพาะด่านที่ระบุ
        await updateDoc(ref, {
            [gameKey]: s,
            ...base,
            lastGame: gameKey,
        });
        return docId;
    }

    // ยังไม่มี → สร้างใหม่ + ใส่คะแนนเฉพาะด่านนี้
    await setDoc(
        ref,
        {
            ...base,
            [gameKey]: s,
            createdAt: now,
            lastGame: gameKey,
        },
        { merge: true }
    );
    return docId;
}

/* ========================= Public APIs ========================= */
/** บันทึก/อัปเดตคะแนนของด่าน 1 (จะสร้างเอกสารใหม่ถ้ายังไม่มี) */
export async function saveGame1Result({ userId, animalId, score, extra = {} }) {
    return upsertScore({ userId, animalId, score, gameKey: "game1", extra });
}
/** บันทึก/อัปเดตคะแนนของด่าน 2 ในเอกสารเดิม */
export async function saveGame2Result({ userId, animalId, score, extra = {} }) {
    return upsertScore({ userId, animalId, score, gameKey: "game2", extra });
}
/** บันทึก/อัปเดตคะแนนของด่าน 3 ในเอกสารเดิม */
export async function saveGame3Result({ userId, animalId, score, extra = {} }) {
    return upsertScore({ userId, animalId, score, gameKey: "game3", extra });
}
/** บันทึก/อัปเดตคะแนนของด่าน 4 ในเอกสารเดิม */
export async function saveGame4Result({ userId, animalId, score, extra = {} }) {
    return upsertScore({ userId, animalId, score, gameKey: "game4", extra });
}

/* ========================= ตัวอย่างการใช้งาน =========================
import { saveGame1Result, saveGame2Result, saveGame3Result, saveGame4Result } from './database.js';

// ด่าน 1
await saveGame1Result({
  userId: 'kid001',
  animalId: 'lion',
  score: 85,
  extra: { source: 'game1', page: location.pathname, ts: Date.now(), animalName: 'สิงโต' }
});

// ด่าน 4
await saveGame4Result({
  userId: 'kid001',
  animalId: 'lion',
  score: 92,
  extra: { source: 'game4', page: location.pathname, ts: Date.now() }
});
====================================================================== */
