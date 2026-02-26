"use client";

import { useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { useBooth } from "../../context/BoothContext";

const BG_URL = "/assets/All%20Done/BG.png";
const IMG = {
  header: "/assets/All%20Done/Header.png",
  title: "/assets/All%20Done/All-Done.png",
  subtitle: "/assets/All%20Done/Your-photo-is-ready!.png",
  character: "/assets/All%20Done/Character.png"
};

function ConfettiBurst() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;

    const ctx = canvas.getContext("2d");
    const colors = ["#ffd166", "#ff6b6b", "#4cc9f0", "#f72585", "#90be6d", "#ff9f1c"];
    const particles = [];

    let width = 0;
    let height = 0;
    let rafId = 0;

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      width = rect.width;
      height = rect.height;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const addBurst = (x, y, count, speedMin, speedMax) => {
      for (let i = 0; i < count; i += 1) {
        const angle = Math.random() * Math.PI * 2;
        const speed = speedMin + Math.random() * (speedMax - speedMin);
        const size = 6 + Math.random() * 6;
        particles.push({
          x,
          y,
          vx: Math.cos(angle) * speed,
          vy: Math.sin(angle) * speed,
          size,
          color: colors[Math.floor(Math.random() * colors.length)],
          rotation: Math.random() * Math.PI,
          spin: (Math.random() * 0.25 + 0.05) * (Math.random() < 0.5 ? -1 : 1),
          life: 120 + Math.random() * 40,
          maxLife: 140,
          round: Math.random() < 0.35
        });
      }
    };

    resize();
    window.addEventListener("resize", resize);

    addBurst(width * 0.2, height * 0.66, 60, 4, 8);
    addBurst(width * 0.5, height * 0.42, 90, 5, 9);
    addBurst(width * 0.8, height * 0.66, 60, 4, 8);

    let last = performance.now();

    const tick = (now) => {
      const dt = Math.min(32, now - last) / 16.67;
      last = now;
      ctx.clearRect(0, 0, width, height);

      const gravity = 0.22;
      const drag = 0.985;

      for (let i = particles.length - 1; i >= 0; i -= 1) {
        const p = particles[i];
        p.vy += gravity * dt;
        p.vx *= Math.pow(drag, dt);
        p.vy *= Math.pow(drag, dt);
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        p.rotation += p.spin * dt;
        p.life -= dt;
        if (p.life <= 0) {
          particles.splice(i, 1);
          continue;
        }

        const alpha = Math.max(0, p.life / p.maxLife);
        ctx.save();
        ctx.translate(p.x, p.y);
        ctx.rotate(p.rotation);
        ctx.globalAlpha = alpha;
        ctx.fillStyle = p.color;
        if (p.round) {
          ctx.beginPath();
          ctx.arc(0, 0, p.size * 0.5, 0, Math.PI * 2);
          ctx.fill();
        } else {
          ctx.fillRect(-p.size / 2, -p.size / 2, p.size, p.size * 0.6);
        }
        ctx.restore();
      }

      if (particles.length > 0) {
        rafId = requestAnimationFrame(tick);
      }
    };

    rafId = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(rafId);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return <canvas ref={canvasRef} className="absolute inset-0 pointer-events-none z-20" />;
}

export default function AllDoneScreen() {
  const router = useRouter();
  const { state } = useBooth();
  const characterId = state.character?.id || "migu";
  const characterLayout = {
    migu: { top: 36, width: 85, left: 54 },
    teddy: { top: 36, width: 70, left: 50 },
    pipper: { top: 36, width: 45, left: 50 },
    tasha: { top: 36, width: 56, left: 50 },
    default: { top: 36, width: 85, left: 54 }
  };
  const layout = characterLayout[characterId] || characterLayout.default;

  useEffect(() => {
    if (!state.shots?.length) router.replace("/capture");
  }, [state.shots, router]);

  return (
    <div className="min-h-screen w-full bg-[#0b2d64] flex items-center justify-center px-4 py-6 kids-font">
      <div className="relative w-full max-w-[520px] aspect-[9/16] overflow-hidden rounded-[32px] shadow-2xl">
        <div
          className="absolute inset-0 bg-cover bg-center"
          style={{ backgroundImage: `url(${BG_URL})` }}
        />

        <img
          src={IMG.header}
          alt=""
          className="absolute left-0 top-0 w-full h-auto"
          draggable="false"
        />

        <ConfettiBurst />

        <img
          src={IMG.title}
          alt="All Done"
          className="absolute left-1/2 top-[24%] w-[44%] -translate-x-1/2"
          draggable="false"
        />
        <img
          src={IMG.subtitle}
          alt="Your photo is ready!"
          className="absolute left-1/2 top-[32%] w-[58%] -translate-x-1/2"
          draggable="false"
        />

        <img
          src={state.composite || IMG.character}
          alt="Character"
          className="absolute -translate-x-1/2"
          style={{ left: `${layout.left}%`, top: `${layout.top}%`, width: `${layout.width}%` }}
          draggable="false"
        />
      </div>
    </div>
  );
}
