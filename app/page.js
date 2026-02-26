"use client";

import { useRouter } from "next/navigation";
import { useBooth } from "../context/BoothContext";

const BG_URL = "/assets/Main%20Screen/BG-new.jpg.jpeg";
const IMG = {
  button: "/assets/Main%20Screen/Button.png",
  logo: "/assets/Main%20Screen/Logo.png"
};

export default function MainScreen() {
  const router = useRouter();
  const { resetAll } = useBooth();

  const onBegin = () => {
    resetAll();
    router.push("/start");
  };

  return (
    <div className="min-h-screen w-full bg-[#2a0b4f] flex items-center justify-center px-4 py-6 kids-font">
      <div className="relative w-full max-w-[520px] aspect-[9/16] overflow-hidden rounded-[32px] shadow-2xl">
        <img
          src={BG_URL}
          alt=""
          className="absolute inset-0 h-full w-full object-cover"
          style={{ objectPosition: "50% 60%" }}
          draggable="false"
        />

        <div className="absolute inset-0">
          <div className="absolute left-1/2 top-[2%] -translate-x-1/2 w-[80%]">
            <img src={IMG.logo} alt="Little Pipers" className="w-full h-auto" draggable="false" />
          </div>

          <div className="absolute left-1/2 bottom-[4%] -translate-x-1/2 w-[58%]">
            <button
              type="button"
              onClick={onBegin}
              className="w-full transition-transform hover:scale-[1.02] active:scale-[0.98]"
              aria-label="Let&apos;s begin"
            >
              <span className="sr-only">Let&apos;s begin</span>
              <img src={IMG.button} alt="" className="w-full h-auto" draggable="false" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
