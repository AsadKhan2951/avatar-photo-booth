import { NextResponse } from "next/server";
import nodemailer from "nodemailer";

export const runtime = "nodejs";

const REQUIRED_ENV = ["SMTP_HOST", "SMTP_PORT", "SMTP_FROM", "SMTP_USER", "SMTP_PASS"];

function isPlaceholder(value) {
  if (!value) return true;
  const v = String(value).toLowerCase();
  return v.includes("your_") || v.includes("changeme") || v.includes("example");
}

function getMissingEnv() {
  return REQUIRED_ENV.filter((key) => isPlaceholder(process.env[key]));
}

function parseDataUrl(dataUrl) {
  const match = dataUrl.match(/^data:(image\/[a-zA-Z0-9.+-]+);base64,(.+)$/);
  if (!match) return null;
  return {
    mime: match[1],
    base64: match[2].replace(/\s/g, "")
  };
}

function safeExt(mime) {
  const ext = mime.split("/")[1] || "png";
  if (ext === "jpeg") return "jpg";
  return ext.replace(/[^a-z0-9]/gi, "");
}

export async function POST(req) {
  try {
    const missing = getMissingEnv();
    if (missing.length) {
      return NextResponse.json(
        { error: `Missing email config: ${missing.join(", ")}` },
        { status: 500 }
      );
    }

    const body = await req.json();
    const to = body?.to;
    const imageData = body?.imageData;
    if (!to || typeof to !== "string") {
      return NextResponse.json({ error: "Missing recipient email." }, { status: 400 });
    }
    if (!imageData || typeof imageData !== "string") {
      return NextResponse.json({ error: "Missing image data." }, { status: 400 });
    }

    const parsed = parseDataUrl(imageData);
    if (!parsed) {
      return NextResponse.json({ error: "Invalid image data format." }, { status: 400 });
    }

    const buffer = Buffer.from(parsed.base64, "base64");
    const ext = safeExt(parsed.mime) || "png";
    const filename = `photo-booth.${ext}`;

    const from = process.env.SMTP_FROM || "";
    if (!/^[^@]+@[^@]+\.[^@]+$/.test(from.replace(/.*<|>.*/g, ""))) {
      return NextResponse.json({ error: "SMTP_FROM must be a valid email address." }, { status: 500 });
    }

    const transporter = nodemailer.createTransport({
      host: process.env.SMTP_HOST,
      port: Number(process.env.SMTP_PORT) || 587,
      secure: process.env.SMTP_SECURE === "true",
      auth: process.env.SMTP_USER
        ? { user: process.env.SMTP_USER, pass: process.env.SMTP_PASS }
        : undefined
    });

    const info = await transporter.sendMail({
      from,
      to,
      subject: body?.subject || "Your Photo Booth Print",
      text: body?.text || "Thanks for visiting! Your photo is attached.",
      attachments: [
        {
          filename,
          content: buffer,
          contentType: parsed.mime
        }
      ]
    });

    return NextResponse.json({ ok: true, messageId: info.messageId });
  } catch (err) {
    const message = err?.message || "Failed to send email.";
    const details = {
      code: err?.code || null,
      responseCode: err?.responseCode || null,
      command: err?.command || null,
      response: err?.response || null
    };
    console.error("Failed to send email", message, details);
    return NextResponse.json({ error: message, details }, { status: 500 });
  }
}
