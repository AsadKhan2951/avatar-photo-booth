import { NextResponse } from "next/server";

export const runtime = "nodejs";

const MODEL = process.env.GEMINI_MODEL || "gemini-2.5-flash-image";
const API_KEY = process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY || "";
const TIMEOUT_MS = Number(process.env.GEMINI_TIMEOUT_MS || 60000);
const AVATAR_PROMPT = process.env.GEMINI_AVATAR_PROMPT
  || "Create a clean, front-facing head-only portrait (forehead to chin). Keep eyes level, no tilt, no shoulders, neutral background, no text.";

function parseDataUrl(dataUrl) {
  const match = dataUrl?.match(/^data:(image\/[a-zA-Z0-9.+-]+);base64,(.+)$/);
  if (!match) return null;
  return { mime: match[1], base64: match[2].replace(/\s/g, "") };
}

function extractImageFromResponse(data) {
  const parts = data?.candidates?.[0]?.content?.parts || [];
  const imagePart = parts.find((p) => p?.inline_data?.data || p?.inlineData?.data);
  const inline = imagePart?.inline_data || imagePart?.inlineData || {};
  const mime = inline?.mime_type || inline?.mimeType || "image/png";
  const base64 = inline?.data || "";
  return { mime, base64 };
}

async function callGemini({ prompt, images, reqId }) {
  const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/${MODEL}:generateContent`;
  const payload = {
    contents: [
      {
        role: "user",
        parts: [
          { text: prompt },
          ...images.map((img) => ({
            inline_data: { mime_type: img.mime, data: img.base64 }
          }))
        ]
      }
    ],
    generationConfig: {
      temperature: 0.6,
      topP: 0.9,
      candidateCount: 1,
      responseModalities: ["IMAGE"]
    }
  };

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), TIMEOUT_MS);
  const startedAt = Date.now();
  const res = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-goog-api-key": API_KEY
    },
    body: JSON.stringify(payload),
    signal: controller.signal
  }).finally(() => clearTimeout(timeout));
  const elapsedMs = Date.now() - startedAt;

  if (!res.ok) {
    const details = await res.text().catch(() => "");
    console.log(`[avatarize:${reqId}] fail status=${res.status} elapsed=${elapsedMs}ms`);
    return {
      ok: false,
      status: res.status,
      elapsedMs,
      details
    };
  }

  const data = await res.json();
  const { mime, base64 } = extractImageFromResponse(data);
  if (!base64) {
    console.log(`[avatarize:${reqId}] fail empty image`);
    return {
      ok: false,
      status: 500,
      elapsedMs,
      details: "No image returned from Gemini."
    };
  }

  console.log(`[avatarize:${reqId}] success elapsed=${elapsedMs}ms`);
  return {
    ok: true,
    mime,
    base64
  };
}

export async function POST(req) {
  try {
    if (!API_KEY) {
      return NextResponse.json({ error: "Missing GEMINI_API_KEY." }, { status: 500 });
    }

    const reqId = Math.random().toString(36).slice(2, 8);
    const startIso = new Date().toISOString();
    console.log(`[avatarize:${reqId}] start ${startIso}`);

    const body = await req.json();
    const imageData = body?.imageData;
    const characterId = body?.characterId || "unknown";
    if (!imageData || typeof imageData !== "string") {
      console.log(`[avatarize:${reqId}] fail missing image data`);
      return NextResponse.json({ error: "Missing image data." }, { status: 400 });
    }

    const parsed = parseDataUrl(imageData);
    if (!parsed) {
      console.log(`[avatarize:${reqId}] fail invalid data url`);
      return NextResponse.json({ error: "Invalid image data format." }, { status: 400 });
    }

    const avatarStep = await callGemini({
      prompt: `${AVATAR_PROMPT}\nCharacter: ${characterId}`,
      images: [parsed],
      reqId
    });
    if (!avatarStep.ok) {
      return NextResponse.json({
        error: "Gemini avatar step failed.",
        status: avatarStep.status,
        elapsedMs: avatarStep.elapsedMs,
        details: avatarStep.details
      }, { status: 500 });
    }

    return NextResponse.json({
      ok: true,
      avatarDataUrl: `data:${avatarStep.mime};base64,${avatarStep.base64}`
    });
  } catch (err) {
    if (err?.name === "AbortError") {
      console.log("[avatarize] timeout");
      return NextResponse.json({ error: "Gemini request timed out." }, { status: 504 });
    }
    const message = err?.message || "Avatarize failed.";
    console.error("Avatarize failed", err);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
