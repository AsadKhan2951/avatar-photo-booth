"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import ScreenCard from "../../components/ScreenCard";
import { GhostButton, PrimaryButton } from "../../components/Button";
import { useBooth } from "../../context/BoothContext";

const STORAGE_KEY = "kids_photo_booth_v1";

function printImage(dataUrl) {
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

  return new Promise((resolve) => {
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
}

export default function Screen6() {
  const router = useRouter();
  const { state, resetAll } = useBooth();
  const [toast, setToast] = useState("");

  const finalImg = useMemo(() => state.enhanced || (state.selectedIndex != null ? state.shots?.[state.selectedIndex] : null), [state]);

  useEffect(() => {
    if (!finalImg) router.replace("/previews");
  }, [finalImg, router]);

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

  const savePrintRecord = async (dataUrl, action) => {
    try {
      await fetch("/api/prints", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          imageData: dataUrl,
          action,
          characterId: state.character?.id || null,
          user: state.user || null
        })
      });
    } catch (err) {
      console.warn("Failed to save print image", err);
    }
  };

  const onPrint = async () => {
    if (!finalImg) return;
    await savePrintRecord(finalImg, "print");
    await printImage(finalImg);
    finishSession();
  };
  const onEmail = async () => {
    if (!finalImg) return;
    await savePrintRecord(finalImg, "email");
    const ok = await sendEmail(finalImg);
    if (ok) finishSession();
  };
  const onBoth = async () => {
    if (finalImg) {
      await savePrintRecord(finalImg, "print_email");
      await printImage(finalImg);
    }
    if (!finalImg) return;
    const ok = await sendEmail(finalImg);
    if (ok) finishSession();
  };

  return (
    <ScreenCard
      screen={7}
      heading="SAVE PHOTO"
      subheading="Choose output method"
      footer={
        <div className="space-y-3">
          <GhostButton onClick={() => router.push("/enhance")}>BACK</GhostButton>
          <button
            type="button"
            onClick={() => { resetAll(); router.push("/"); }}
            className="w-full text-xs text-slate-500 hover:text-slate-700"
          >
            Start new session
          </button>
        </div>
      }
    >
      <div className="px-6 pb-6">
        <div className="rounded-3xl border-2 border-slate-300 bg-white p-3">
          {finalImg ? (
            <img src={finalImg} alt="Final" className="w-full aspect-[3/4] object-cover rounded-2xl" />
          ) : null}
        </div>

        <div className="mt-5 space-y-3">
          <button type="button" onClick={onPrint} className="k-btn k-btn-ghost">
            🖨️ PRINT PHOTO
          </button>
          <button type="button" onClick={onEmail} className="k-btn k-btn-ghost">
            📧 EMAIL PHOTO
          </button>
          <PrimaryButton onClick={onBoth}>
            🖨️ + 📧 PRINT & EMAIL
          </PrimaryButton>
        </div>

        {toast ? (
          <div className="mt-4 rounded-2xl border border-sky-200 bg-sky-50 px-4 py-3 text-sm text-sky-800">
            {toast}
          </div>
        ) : null}
      </div>
    </ScreenCard>
  );
}
