"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import Countdown from "../../components/Countdown";
import { useBooth } from "../../context/BoothContext";

export default function CountdownScreen() {
  const router = useRouter();
  const { state } = useBooth();

  useEffect(() => {
    if (!state.character) router.replace("/character");
  }, [state.character, router]);

  return (
    <div className="min-h-screen w-full bg-[#2a0b4f] flex items-center justify-center px-4 py-6 kids-font">
      <div className="relative w-full max-w-[520px] aspect-[9/16] overflow-hidden rounded-[32px] shadow-2xl">
        <Countdown seconds={3} onDone={() => router.replace("/capture")} />
      </div>
    </div>
  );
}
