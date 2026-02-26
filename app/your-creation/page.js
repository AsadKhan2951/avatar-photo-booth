"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { useBooth } from "../../context/BoothContext";

const BG_URL = "/assets/Your%20Creation/Your-Creation-BG.jpg";
const PRINT_HEADER_BG_URL = "/assets/Start%20Screen/Layer-Top-Print.jpg";
const PRINT_FOOTER_BG_URL = "/assets/Start%20Screen/Layer-Bottom-Print.jpg";
const IMG = {
  character: "/assets/Your%20Creation/Character.png",
  print: "/assets/Your%20Creation/3.png",
  email: "/assets/Your%20Creation/2.png",
  both: "/assets/Your%20Creation/1.png",
  heartLeft: "/assets/Your%20Creation/Heart-Left.png",
  heartRight: "/assets/Your%20Creation/Heart-Right.png",
  star: "/assets/Your%20Creation/Star.png",
  labelShape: "/assets/Your%20Creation/shape.png",
  arrow: "/assets/Start%20Screen/Arrow.png"
};
const LABELS = {
  migu: "/assets/Your%20Creation/MIGU.png",
  teddy: "/assets/Your%20Creation/TEDDY.png",
  pipper: "/assets/Your%20Creation/PIPER.png",
  tasha: "/assets/Tasha/Tasha.png"
};
const CHARACTER_NAMES = {
  migu: "MIGU",
  teddy: "GLUCO TEDDY",
  pipper: "PIED PIPER",
  tasha: "TASHA"
};
const CARD_THEME = {
  migu: { bg: "linear-gradient(180deg, #1d72ff 0%, #0f53cc 100%)" },
  teddy: { bg: "linear-gradient(180deg, #f1943b 0%, #c86a1d 100%)" },
  pipper: { bg: "linear-gradient(180deg, #ffd25a 0%, #f1a52b 100%)" },
  tasha: { bg: "linear-gradient(180deg, #ff58bf 0%, #d9349f 100%)" },
  default: { bg: "linear-gradient(180deg, #1d72ff 0%, #0f53cc 100%)" }
};
const UI_CARD_IMAGE = {
  migu: { top: 28, width: 84 },
  teddy: { top: 30, width: 76 },
  pipper: { top: 40, width: 56 },
  tasha: { top: 32, width: 62 },
  default: { top: 28, width: 84 }
};
const PRINT_CHARACTER_LAYOUT = {
  migu: { top: 0.26, width: 0.86, height: 0.62, zoom: 1.1, yShift: 0.07, labelClearance: 1.02 },
  teddy: { top: 0.25, width: 0.8, height: 0.62, zoom: 1.08, yShift: 0.055, labelClearance: 0.86 },
  pipper: { top: 0.245, width: 0.78, height: 0.62, zoom: 1.02, yShift: 0.02, labelClearance: 0.8 },
  tasha: { top: 0.25, width: 0.8, height: 0.62, zoom: 1.04, yShift: 0.05, labelClearance: 0.9 },
  default: { top: 0.26, width: 0.84, height: 0.62, zoom: 1.08, yShift: 0.07, labelClearance: 1.02 }
};
const PRINT_LABEL_LAYOUT = {
  migu: { width: 0.56, y: 0.18 },
  teddy: { width: 0.5, y: 0.17 },
  pipper: { width: 0.5, y: 0.17 },
  tasha: { width: 0.5, y: 0.18 },
  default: { width: 0.52, y: 0.17 }
};
const PRINT_SHAPE_LAYOUT = {
  width: 0.55,
  heightScale: 0.68,
  y: 0.075
};
const STORAGE_KEY = "kids_photo_booth_v1";
const PRINT_FOOTER_FONT = "Cardust";

function ordinalSuffix(day) {
  const mod10 = day % 10;
  const mod100 = day % 100;
  if (mod10 === 1 && mod100 !== 11) return "st";
  if (mod10 === 2 && mod100 !== 12) return "nd";
  if (mod10 === 3 && mod100 !== 13) return "rd";
  return "th";
}

function formatPrintDate(date = new Date()) {
  const day = date.getDate();
  const suffix = ordinalSuffix(day);
  const month = date.toLocaleString("en-US", { month: "long" }).toUpperCase();
  const year = date.getFullYear();
  return `${day}${suffix} - ${month} - ${year}`;
}

function drawCover(ctx, img, x, y, width, height, options = {}) {
  const { alignX = 0.5, alignY = 0.5 } = options;
  const scale = Math.max(width / img.width, height / img.height);
  const drawW = img.width * scale;
  const drawH = img.height * scale;
  const dx = x - (drawW - width) * alignX;
  const dy = y - (drawH - height) * alignY;
  ctx.drawImage(img, dx, dy, drawW, drawH);
}

function fitText(ctx, text, maxWidth, maxSize, minSize, fontFamily, weight = 900) {
  let size = maxSize;
  while (size >= minSize) {
    ctx.font = `${weight} ${size}px ${fontFamily}`;
    if (ctx.measureText(text).width <= maxWidth) return size;
    size -= 2;
  }
  return minSize;
}

function getOpaqueBounds(img) {
  const probe = document.createElement("canvas");
  probe.width = img.width;
  probe.height = img.height;
  const pctx = probe.getContext("2d");
  pctx.drawImage(img, 0, 0);
  const data = pctx.getImageData(0, 0, probe.width, probe.height).data;

  let minX = probe.width;
  let minY = probe.height;
  let maxX = -1;
  let maxY = -1;

  for (let y = 0; y < probe.height; y += 1) {
    for (let x = 0; x < probe.width; x += 1) {
      const a = data[(y * probe.width + x) * 4 + 3];
      if (a <= 10) continue;
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
    }
  }

  if (maxX < 0 || maxY < 0) {
    return { x: 0, y: 0, width: img.width, height: img.height };
  }
  return {
    x: minX,
    y: minY,
    width: Math.max(1, maxX - minX + 1),
    height: Math.max(1, maxY - minY + 1)
  };
}

async function ensurePrintFonts() {
  if (typeof document === "undefined" || !document.fonts?.load) return;
  try {
    const hasCardust = Array.from(document.fonts).some((f) => f.family === PRINT_FOOTER_FONT);
    if (!hasCardust) {
      const face = new FontFace(PRINT_FOOTER_FONT, "url(/Cardust.otf)");
      await face.load();
      document.fonts.add(face);
    }
    await Promise.all([
      document.fonts.load(`900 64px "${PRINT_FOOTER_FONT}"`),
      document.fonts.load(`800 44px "${PRINT_FOOTER_FONT}"`),
      document.fonts.ready
    ]);
  } catch {}
}

export default function YourCreationScreen() {
  const router = useRouter();
  const { state, resetAll } = useBooth();
  const [toast, setToast] = useState("");
  const [printing, setPrinting] = useState(false);

  const finalImg = useMemo(() => {
    if (state.composite) return state.composite;
    if (state.enhanced) return state.enhanced;
    return state.shots?.[0] || null;
  }, [state.composite, state.enhanced, state.shots]);
  const characterId = state.character?.id || "migu";
  const labelImg = LABELS[characterId] || LABELS.migu;
  const cardTheme = CARD_THEME[characterId] || CARD_THEME.default;
  const cardImage = UI_CARD_IMAGE[characterId] || UI_CARD_IMAGE.default;

  useEffect(() => {
    if (!state.shots?.length) router.replace("/capture");
  }, [state.shots, router]);

  const finishSession = () => {
    try {
      if (typeof window !== "undefined") {
        localStorage.removeItem(STORAGE_KEY);
      }
    } catch {}
    resetAll();
    if (typeof window !== "undefined") {
      window.location.replace("/");
      return;
    }
    router.replace("/");
  };

  const sendEmail = async (imageData) => {
    const email = String(state.user?.email || "").trim();
    if (!email) {
      setToast("Email address missing. Please enter your email.");
      setTimeout(() => setToast(""), 2200);
      return false;
    }
    try {
      const res = await fetch("/api/email", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          to: email,
          imageData,
          subject: "Your Photo Booth Print",
          text: "Thanks for visiting! Your photo is attached."
        })
      });
      if (!res.ok) {
        const payload = await res.json().catch(() => ({}));
        setToast(payload.error || "Email failed. Please try again.");
        setTimeout(() => setToast(""), 2200);
        return false;
      }
      setToast("Email sent successfully.");
      setTimeout(() => setToast(""), 2200);
      return true;
    } catch (err) {
      console.warn("Email send failed", err);
      setToast("Email failed. Please try again.");
      setTimeout(() => setToast(""), 2200);
      return false;
    }
  };

  const savePrintRecord = async (dataUrl, action) => {
    try {
      await fetch("/api/prints", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          imageData: dataUrl,
          action,
          characterId,
          user: state.user || null
        })
      });
    } catch (err) {
      console.warn("Failed to save print image", err);
    }
  };

  const loadImage = (src) =>
    new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = src;
    });

  const composePrintImage = async (dataUrl, selectedId) => {
    await ensurePrintFonts();
    const [headerBgImg, footerBgImg, photoImg, shapeImg, labelImgObj, arrowImg, starImg] = await Promise.all([
      loadImage(PRINT_HEADER_BG_URL),
      loadImage(PRINT_FOOTER_BG_URL),
      loadImage(dataUrl),
      loadImage(IMG.labelShape),
      loadImage(LABELS[selectedId] || LABELS.migu),
      loadImage(IMG.arrow),
      loadImage(IMG.star)
    ]);
    const canvas = document.createElement("canvas");
    canvas.width = 1200;
    canvas.height = 1800;
    const ctx = canvas.getContext("2d");
    const headerH = Math.round(canvas.width * (headerBgImg.height / headerBgImg.width));
    const footerH = Math.round(canvas.height * 0.19);
    const middleY = headerH;
    const middleH = canvas.height - headerH - footerH;
    const centerX = canvas.width / 2;

    // Keep top strip fully visible without cutting.
    ctx.drawImage(headerBgImg, 0, 0, canvas.width, headerH);
    drawCover(ctx, footerBgImg, 0, canvas.height - footerH, canvas.width, footerH, { alignY: 0 });

    const middleGrad = ctx.createLinearGradient(0, middleY, 0, middleY + middleH);
    middleGrad.addColorStop(0, "#1f78ff");
    middleGrad.addColorStop(1, "#0f59dc");
    ctx.fillStyle = middleGrad;
    // Slight overlap removes any visible seam between header and body.
    ctx.fillRect(0, middleY - 1, canvas.width, middleH + 2);

    const glow = ctx.createRadialGradient(
      centerX,
      middleY + middleH * 0.42,
      10,
      centerX,
      middleY + middleH * 0.42,
      canvas.width * 0.52
    );
    glow.addColorStop(0, "rgba(255,255,255,0.22)");
    glow.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = glow;
    ctx.fillRect(0, middleY, canvas.width, middleH);

    const shapeW = canvas.width * PRINT_SHAPE_LAYOUT.width;
    const shapeH = (shapeImg.height / shapeImg.width) * shapeW * PRINT_SHAPE_LAYOUT.heightScale;
    const shapeX = centerX - shapeW / 2;
    const shapeY = middleY + middleH * PRINT_SHAPE_LAYOUT.y;
    ctx.drawImage(shapeImg, shapeX, shapeY, shapeW, shapeH);

    const labelLayout = PRINT_LABEL_LAYOUT[selectedId] || PRINT_LABEL_LAYOUT.default;
    const labelW = shapeW * labelLayout.width;
    const labelH = (labelImgObj.height / labelImgObj.width) * labelW;
    const labelX = centerX - labelW / 2;
    const labelY = shapeY + shapeH * labelLayout.y;
    ctx.drawImage(labelImgObj, labelX, labelY, labelW, labelH);

    const charLayout = PRINT_CHARACTER_LAYOUT[selectedId] || PRINT_CHARACTER_LAYOUT.default;
    const opaque = getOpaqueBounds(photoImg);
    const charAreaY = middleY + middleH * charLayout.top;
    const charAreaH = middleH * charLayout.height;
    const charAreaW = canvas.width * charLayout.width;
    const baseScale = Math.min(charAreaW / opaque.width, charAreaH / opaque.height);
    const zoom = charLayout.zoom || 1;
    const charScale = baseScale * zoom;
    const charW = opaque.width * charScale;
    const charH = opaque.height * charScale;
    const charX = centerX - charW / 2;
    const charYRaw = charAreaY + (charAreaH - charH) / 2 + middleH * (charLayout.yShift || 0);
    const minCharY = labelY + labelH * (charLayout.labelClearance || 1.02);
    const charY = Math.max(charYRaw, minCharY);
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, middleY, canvas.width, middleH);
    ctx.clip();
    ctx.drawImage(photoImg, opaque.x, opaque.y, opaque.width, opaque.height, charX, charY, charW, charH);
    ctx.restore();

    const footerY = canvas.height - footerH;

    const displayName = String(state.user?.name || "").trim().toUpperCase() || "YOUNG PIPER";
    const dateText = formatPrintDate();
    const titleFamily = `"${PRINT_FOOTER_FONT}", "Baloo 2", "Arial Black", sans-serif`;

    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    const dateSize = fitText(ctx, dateText, canvas.width * 0.72, 52, 28, titleFamily, 800);
    ctx.font = `800 ${dateSize}px ${titleFamily}`;
    ctx.lineWidth = 6;
    ctx.strokeStyle = "#ffffff";
    ctx.fillStyle = "#5b26cf";
    ctx.strokeText(dateText, centerX, footerY + footerH * 0.22);
    ctx.fillText(dateText, centerX, footerY + footerH * 0.22);

    const nameSize = fitText(ctx, displayName, canvas.width * 0.8, 120, 54, titleFamily, 900);
    ctx.font = `900 ${nameSize}px ${titleFamily}`;
    ctx.lineWidth = Math.max(5, Math.round(nameSize * 0.075));
    ctx.strokeStyle = "#ffd52a";
    ctx.fillStyle = "#ef2f2f";
    ctx.strokeText(displayName, centerX, footerY + footerH * 0.58);
    ctx.fillText(displayName, centerX, footerY + footerH * 0.58);

    const subline = `WITH LOVE ${CHARACTER_NAMES[selectedId] || "PIED PIPER"}`;
    const sublineSize = fitText(ctx, subline, canvas.width * 0.66, 50, 24, titleFamily, 800);
    ctx.font = `800 ${sublineSize}px ${titleFamily}`;
    ctx.lineWidth = 4;
    ctx.strokeStyle = "#ffffff";
    ctx.fillStyle = "#7a1acb";
    ctx.strokeText(subline, centerX, footerY + footerH * 0.86);
    ctx.fillText(subline, centerX, footerY + footerH * 0.86);

    const arrowW = canvas.width * 0.09;
    const arrowH = (arrowImg.height / arrowImg.width) * arrowW;
    ctx.drawImage(arrowImg, canvas.width * 0.08, footerY + footerH * 0.34, arrowW, arrowH);

    const starW = canvas.width * 0.08;
    const starH = (starImg.height / starImg.width) * starW;
    ctx.drawImage(starImg, canvas.width * 0.78, footerY + footerH * 0.64, starW, starH);

    ctx.font = `900 ${Math.round(footerH * 0.22)}px ${titleFamily}`;
    ctx.fillStyle = "#d011b1";
    ctx.fillText("♪", canvas.width * 0.83, footerY + footerH * 0.24);

    return canvas.toDataURL("image/png");
  };

  const printImage = (dataUrl) => new Promise((resolve) => {
    const originalTitle = document.title;
    document.title = "";
    const iframe = document.createElement("iframe");
    iframe.setAttribute("aria-hidden", "true");
    iframe.style.position = "fixed";
    iframe.style.right = "0";
    iframe.style.bottom = "0";
    iframe.style.width = "0";
    iframe.style.height = "0";
    iframe.style.border = "0";
    iframe.style.opacity = "0";
    document.body.appendChild(iframe);

    const doc = iframe.contentWindow?.document;
    if (!doc) {
      iframe.remove();
      resolve();
      return;
    }

    doc.open();
    doc.write(`
      <!doctype html>
      <html>
        <head>
          <title></title>
          <style>
            @page { margin: 0; size: 5in 7in; }
            html, body { margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; }
            body { background: #0b2d64; }
            img { width: 96%; height: 96%; object-fit: contain; display: block; margin: 2% auto; }
            @media print {
              body { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
            }
          </style>
        </head>
        <body>
          <img id="print-img" src="${dataUrl}" />
        </body>
      </html>
    `);
    doc.close();

    let done = false;
    const cleanup = () => {
      if (done) return;
      done = true;
      document.title = originalTitle;
      iframe.remove();
      resolve();
    };

    const triggerPrint = () => {
      const w = iframe.contentWindow;
      if (!w) {
        cleanup();
        return;
      }
      const onAfterPrint = () => cleanup();
      w.addEventListener("afterprint", onAfterPrint, { once: true });
      w.focus();
      w.print();
      setTimeout(cleanup, 6000);
    };

    const img = doc.getElementById("print-img");
    if (img) {
      img.onload = triggerPrint;
      img.onerror = cleanup;
    } else {
      triggerPrint();
    }
  });

  const onPrint = async () => {
    if (!finalImg || printing) return;
    setPrinting(true);
    try {
      const composed = await composePrintImage(finalImg, characterId);
      await savePrintRecord(composed, "print");
      await printImage(composed);
    } finally {
      setPrinting(false);
    }
    finishSession();
  };

  const onEmail = async () => {
    if (!finalImg || printing) return;
    setPrinting(true);
    try {
      const composed = await composePrintImage(finalImg, characterId);
      await savePrintRecord(composed, "email");
      const ok = await sendEmail(composed);
      if (ok) finishSession();
    } finally {
      setPrinting(false);
    }
  };

  const onBoth = async () => {
    if (!finalImg || printing) return;
    setPrinting(true);
    let composed = null;
    try {
      composed = await composePrintImage(finalImg, characterId);
      await savePrintRecord(composed, "print_email");
      await printImage(composed);
    } finally {
      setPrinting(false);
    }
    if (!composed) return;
    const ok = await sendEmail(composed);
    if (ok) finishSession();
  };

  return (
    <div className="min-h-screen w-full bg-[#0b2d64] flex items-center justify-center px-4 py-6 kids-font">
      <div className="relative w-full max-w-[520px] aspect-[9/16] overflow-hidden rounded-[32px] shadow-2xl">
        <div
          className="absolute inset-0 bg-cover bg-center"
          style={{ backgroundImage: `url(${BG_URL})` }}
        />
        <img
          src={IMG.heartLeft}
          alt=""
          className="absolute left-[18%] top-[7%] w-[12%] h-auto"
          draggable="false"
        />
        <img
          src={IMG.star}
          alt=""
          className="absolute left-1/2 top-[5.5%] w-[10%] h-auto -translate-x-1/2"
          draggable="false"
        />
        <img
          src={IMG.heartRight}
          alt=""
          className="absolute right-[18%] top-[7%] w-[12%] h-auto"
          draggable="false"
        />

        <div className="absolute left-1/2 top-[14%] w-[66%] -translate-x-1/2">
          <div
            className="relative w-full aspect-[3/4] rounded-[28px] border-[6px] border-[#7b33ff] shadow-[0_18px_30px_rgba(0,0,0,0.28)]"
            style={{ background: cardTheme.bg }}
          >
            <div className="absolute left-1/2 top-[6%] w-[84%] -translate-x-1/2">
              <div className="relative w-full">
                <img src={IMG.labelShape} alt="" className="w-full h-auto" draggable="false" />
                <img
                  src={labelImg}
                  alt=""
                  className="absolute left-1/2 top-[18%] w-[74%] -translate-x-1/2"
                  draggable="false"
                />
              </div>
            </div>
            <img
              src={finalImg || IMG.character}
              alt="Character"
              className="absolute left-1/2 -translate-x-1/2"
              style={{ top: `${cardImage.top}%`, width: `${cardImage.width}%` }}
              draggable="false"
            />
          </div>
        </div>

        <div className="absolute left-1/2 bottom-[4%] w-[42%] -translate-x-1/2 flex flex-col items-center gap-[clamp(10px,2vh,16px)]">
          <button
            type="button"
            onClick={onPrint}
            className="w-full transition-transform active:scale-[0.98] hover:scale-[1.03]"
            aria-label="Print"
          >
            <img src={IMG.print} alt="Print" className="w-full h-auto" draggable="false" />
          </button>
          <button
            type="button"
            onClick={onEmail}
            className="w-full transition-transform active:scale-[0.98] hover:scale-[1.03]"
            aria-label="Email"
          >
            <img src={IMG.email} alt="Email" className="w-full h-auto" draggable="false" />
          </button>
          <button
            type="button"
            onClick={onBoth}
            className="w-[112%] transition-transform active:scale-[0.98] hover:scale-[1.03]"
            aria-label="Do both"
          >
            <img src={IMG.both} alt="Do both" className="w-full h-auto" draggable="false" />
          </button>
        </div>

        {toast ? (
          <div className="absolute left-1/2 bottom-[20%] -translate-x-1/2 rounded-xl bg-white/90 px-3 py-1 text-[11px] font-semibold text-slate-700">
            {toast}
          </div>
        ) : null}
      </div>
    </div>
  );
}
