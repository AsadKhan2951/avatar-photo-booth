"use client";

import { createContext, useContext, useEffect, useMemo, useState } from "react";

const BoothContext = createContext(null);

const STORAGE_KEY = "kids_photo_booth_v1";

const emptyState = {
  user: { name: "", schoolName: "", age: "", gender: "", email: "", phone: "" },
  character: null, // {id,name}
  shots: [], // dataURLs
  selectedIndex: null,
  enhanced: null, // dataURL (mock)
  avatar: null, // generated avatar face dataURL
  composite: null, // final character image path/dataURL
  poseLock: false // keep same pose across "try another photo" recaptures
};

export function BoothProvider({ children }) {
  const [state, setState] = useState(emptyState);

  // restore
  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) setState(JSON.parse(raw));
    } catch {}
  }, []);

  // persist
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch {}
  }, [state]);

  const api = useMemo(() => ({
    state,
    setUser: (user) => setState((s) => ({ ...s, user: { ...s.user, ...user } })),
    setCharacter: (character) => setState((s) => ({
      ...s,
      character,
      avatar: null,
      composite: null,
      poseLock: false
    })),
    setShots: (shots) => setState((s) => ({
      ...s,
      shots,
      selectedIndex: null,
      enhanced: null,
      avatar: null,
      composite: s.poseLock ? s.composite : null
    })),
    selectShot: (index) => setState((s) => ({ ...s, selectedIndex: index, enhanced: null })),
    setEnhanced: (enhanced) => setState((s) => ({ ...s, enhanced })),
    setAvatar: (avatar) => setState((s) => ({ ...s, avatar })),
    setComposite: (composite) => setState((s) => ({ ...s, composite })),
    setPoseLock: (poseLock) => setState((s) => ({ ...s, poseLock })),
    resetAll: () => setState(emptyState)
  }), [state]);

  return <BoothContext.Provider value={api}>{children}</BoothContext.Provider>;
}

export function useBooth() {
  const ctx = useContext(BoothContext);
  if (!ctx) throw new Error("useBooth must be used inside BoothProvider");
  return ctx;
}
