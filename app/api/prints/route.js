import { NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";
import { v2 as cloudinary } from "cloudinary";

export const runtime = "nodejs";

const STORAGE_DIR = path.join(process.cwd(), "storage", "prints");
const LOG_PATH = path.join(STORAGE_DIR, "log.jsonl");

const CLOUD_NAME = process.env.CLOUDINARY_CLOUD_NAME;
const CLOUD_KEY = process.env.CLOUDINARY_API_KEY;
const CLOUD_SECRET = process.env.CLOUDINARY_API_SECRET;
const CLOUD_FOLDER = process.env.CLOUDINARY_FOLDER || "photo-booth";
const USE_CLOUDINARY = Boolean(CLOUD_NAME && CLOUD_KEY && CLOUD_SECRET) || Boolean(process.env.CLOUDINARY_URL);

function isPlaceholder(value) {
  if (!value) return true;
  const v = String(value).toLowerCase();
  return v.includes("your_") || v.includes("changeme") || v.includes("example");
}

const CLOUDINARY_READY = !isPlaceholder(CLOUD_NAME)
  && !isPlaceholder(CLOUD_KEY)
  && !isPlaceholder(CLOUD_SECRET);

if (USE_CLOUDINARY && CLOUDINARY_READY) {
  cloudinary.config({
    cloud_name: CLOUD_NAME,
    api_key: CLOUD_KEY,
    api_secret: CLOUD_SECRET,
    secure: true
  });
}

function safeExt(mime) {
  const ext = mime.split("/")[1] || "png";
  if (ext === "jpeg") return "jpg";
  return ext.replace(/[^a-z0-9]/gi, "");
}

export async function POST(req) {
  try {
    const body = await req.json();
    const imageData = body?.imageData;
    if (!imageData || typeof imageData !== "string") {
      return NextResponse.json({ error: "Missing image data." }, { status: 400 });
    }

    const match = imageData.match(/^data:(image\/[a-zA-Z0-9.+-]+);base64,(.+)$/);
    if (!match) {
      return NextResponse.json({ error: "Invalid image data format." }, { status: 400 });
    }

    const mime = match[1];
    const base64 = match[2].replace(/\s/g, "");
    const buffer = Buffer.from(base64, "base64");
    const ext = safeExt(mime) || "png";

    if (USE_CLOUDINARY && CLOUDINARY_READY) {
      const action = body?.action || "print";
      const characterId = body?.characterId || "unknown";
      const upload = await cloudinary.uploader.upload(imageData, {
        folder: CLOUD_FOLDER,
        resource_type: "image",
        tags: ["photo-booth", action, characterId],
        context: {
          action,
          characterId
        }
      });
      return NextResponse.json({
        ok: true,
        cloudinary: {
          publicId: upload.public_id,
          url: upload.secure_url
        }
      });
    }

    await fs.mkdir(STORAGE_DIR, { recursive: true });
    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    const rand = Math.random().toString(36).slice(2, 8);
    const filename = `${stamp}_${rand}.${ext}`;
    const filePath = path.join(STORAGE_DIR, filename);
    await fs.writeFile(filePath, buffer);

    const logEntry = {
      time: new Date().toISOString(),
      file: filename,
      size: buffer.length,
      action: body?.action || "print",
      characterId: body?.characterId || null,
      user: body?.user || null
    };
    await fs.appendFile(LOG_PATH, `${JSON.stringify(logEntry)}\n`);

    return NextResponse.json({ ok: true, file: filename });
  } catch (err) {
    console.error("Failed to save print image", err);
    return NextResponse.json({ error: "Failed to save image." }, { status: 500 });
  }
}
