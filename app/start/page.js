"use client";

import { useRouter } from "next/navigation";
import { useBooth } from "../../context/BoothContext";
import { useMemo, useState } from "react";

const IMG = {
  button: "/assets/Start%20Screen/Select.png",
  layerTop: "/assets/Start%20Screen/Layer-Top.jpg",
  layerBottom: "/assets/Start%20Screen/Layer-Bottom.jpg"
};

export default function StartScreen() {
  const router = useRouter();
  const { state, setUser, resetAll } = useBooth();
  const [form, setForm] = useState({
    name: state.user?.name || "",
    schoolName: state.user?.schoolName || "",
    age: state.user?.age || "",
    gender: state.user?.gender || "",
    email: state.user?.email || "",
    phone: state.user?.phone || ""
  });
  const [error, setError] = useState("");

  const valid = useMemo(() => {
    if (!form.gender) return false;
    if (!form.email.trim()) return false;
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(form.email.trim())) return false;
    if (!/^\d{11}$/.test(String(form.phone).trim())) return false;
    return true;
  }, [form]);

  const onNext = () => {
    if (!valid) {
      setError("Please select gender, and enter a valid email + 11-digit phone number.");
      return;
    }
    setError("");
    setUser(form);
    router.push("/character");
  };

  // On-screen keyboard removed; use physical keyboard.

  return (
    <div className="min-h-screen w-full bg-[#f6c11e] flex items-center justify-center px-4 py-6 kids-font">
      <div className="relative w-full max-w-[735px] aspect-[9/16] overflow-hidden rounded-[32px] shadow-2xl">
        <div className="absolute inset-0 bg-[#f6c11e]" />

        <div className="absolute inset-0">
          <img
            src={IMG.layerTop}
            alt=""
            className="absolute left-0 top-0 w-full h-auto"
            draggable="false"
          />
          <img
            src={IMG.layerBottom}
            alt=""
            className="absolute left-0 bottom-0 w-full h-auto"
            draggable="false"
          />
          <div className="absolute left-1/2 top-[25%] bottom-[18%] -translate-x-1/2 w-[90%] flex flex-col items-center">
            <div className="text-center">
              <div className="text-[#a424c7] font-extrabold tracking-wide text-[clamp(16px,2.7vw,28px)]">
                ENTER DETAILS
              </div>
              <div className="text-[#a424c7] font-semibold text-[clamp(10px,1.9vw,17px)]">
                Fill in your information
              </div>
            </div>

            <div
              className="w-full flex flex-col"
              style={{ gap: "clamp(5px, 1.2vw, 10px)" }}
            >
              <label className="text-[#a424c7] font-bold tracking-wide text-[clamp(10px,1.5vw,14px)]">
                NAME
                <input
                  className="mt-1 w-full rounded-[12px] border-[2.5px] border-white bg-[#ffe38c] px-4 text-[#a424c7] placeholder:text-[#2d2bb8]/70 outline-none"
                  style={{ height: "clamp(26px, 4.3vw, 38px)" }}
                  value={form.name}
                  placeholder="Enter your name"
                  autoComplete="new-password"
                  name="name-field"
                  id="name-field"
                  autoCorrect="off"
                  autoCapitalize="off"
                  spellCheck={false}
                  onChange={(e) => { setError(""); setForm((s) => ({ ...s, name: e.target.value })); }}
                  onFocus={() => setError("")}
                />
              </label>

              <label className="text-[#a424c7] font-bold tracking-wide text-[clamp(10px,1.5vw,14px)]">
                AGE
                <input
                  className="mt-1 w-full rounded-[12px] border-[2.5px] border-white bg-[#ffe38c] px-4 text-[#a424c7] placeholder:text-[#2d2bb8]/70 outline-none"
                  style={{ height: "clamp(26px, 4.3vw, 38px)" }}
                  inputMode="numeric"
                  value={form.age}
                  placeholder="Enter your age"
                  autoComplete="new-password"
                  name="age-field"
                  id="age-field"
                  autoCorrect="off"
                  autoCapitalize="off"
                  spellCheck={false}
                  onChange={(e) => { setError(""); setForm((s) => ({ ...s, age: e.target.value })); }}
                  onFocus={() => setError("")}
                />
              </label>

              <label className="text-[#a424c7] font-bold tracking-wide text-[clamp(10px,1.5vw,14px)]">
                SCHOOL NAME
                <input
                  className="mt-1 w-full rounded-[12px] border-[2.5px] border-white bg-[#ffe38c] px-4 text-[#a424c7] placeholder:text-[#2d2bb8]/70 outline-none"
                  style={{ height: "clamp(26px, 4.3vw, 38px)" }}
                  value={form.schoolName}
                  placeholder="Enter your school name"
                  autoComplete="new-password"
                  name="school-name-field"
                  id="school-name-field"
                  autoCorrect="off"
                  autoCapitalize="off"
                  spellCheck={false}
                  onChange={(e) => { setError(""); setForm((s) => ({ ...s, schoolName: e.target.value })); }}
                  onFocus={() => setError("")}
                />
              </label>

              <label className="text-[#a424c7] font-bold tracking-wide text-[clamp(10px,1.5vw,14px)]">
                EMAIL ADDRESS
                <input
                  className="mt-1 w-full rounded-[12px] border-[2.5px] border-white bg-[#ffe38c] px-4 text-[#a424c7] placeholder:text-[#2d2bb8]/70 outline-none"
                  style={{ height: "clamp(26px, 4.3vw, 38px)" }}
                  value={form.email}
                  placeholder="yourname@example.com"
                  autoComplete="email"
                  name="email-field"
                  id="email-field"
                  autoCorrect="off"
                  autoCapitalize="off"
                  spellCheck={false}
                  onChange={(e) => {
                    setError("");
                    setForm((s) => ({ ...s, email: e.target.value }));
                  }}
                  onFocus={() => setError("")}
                />
              </label>

              <label className="text-[#a424c7] font-bold tracking-wide text-[clamp(10px,1.5vw,14px)]">
                GENDER
                <select
                  className="mt-1 w-full rounded-[12px] border-[2.5px] border-white bg-[#ffe38c] px-4 text-[#a424c7] outline-none"
                  style={{ height: "clamp(26px, 4.3vw, 38px)" }}
                  value={form.gender}
                  name="gender-field"
                  id="gender-field"
                  onChange={(e) => {
                    setError("");
                    setForm((s) => ({ ...s, gender: e.target.value }));
                  }}
                  onFocus={() => setError("")}
                >
                  <option value="">Select gender</option>
                  <option value="male">Male</option>
                  <option value="female">Female</option>
                </select>
              </label>

              <label className="text-[#a424c7] font-bold tracking-wide text-[clamp(10px,1.5vw,14px)]">
                CONTACT NUMBER
                <input
                  className="mt-1 w-full rounded-[12px] border-[2.5px] border-white bg-[#ffe38c] px-4 text-[#a424c7] placeholder:text-[#2d2bb8]/70 outline-none"
                  style={{ height: "clamp(26px, 4.3vw, 38px)" }}
                  value={form.phone}
                  placeholder="11-digit number"
                  inputMode="numeric"
                  pattern="\d*"
                  autoComplete="new-password"
                  name="phone-field"
                  id="phone-field"
                  autoCorrect="off"
                  autoCapitalize="off"
                  spellCheck={false}
                  onChange={(e) => {
                    setError("");
                    const digits = e.target.value.replace(/\D/g, "").slice(0, 11);
                    setForm((s) => ({ ...s, phone: digits }));
                  }}
                  onFocus={() => setError("")}
                />
              </label>
            </div>

            <div
              className="mt-2 w-[26%] transition-opacity"
            >
              <button
                type="button"
                onClick={onNext}
                className="w-full transition-transform active:scale-[0.98] hover:scale-[1.02]"
                aria-label="Next"
              >
                <img src={IMG.button} alt="" className="w-full h-auto" draggable="false" />
              </button>
            
            </div>
            {error ? (
              <div className="mt-2 rounded-lg bg-white/70 px-3 py-1 text-center text-xs font-semibold text-[#a424c7]">
                {error}
              </div>
            ) : null}
          </div>

          

          <button
            type="button"
            onClick={resetAll}
            className="absolute right-[4%] top-[4%] text-[0px]"
            aria-label="Reset session"
          >
            Reset
          </button>
        </div>
      </div>
    </div>
  );
}


