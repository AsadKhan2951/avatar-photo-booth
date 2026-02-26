import "./globals.css";
import { BoothProvider } from "../context/BoothContext";
import PageTransition from "../components/PageTransition";

export const metadata = {
  title: "Kids Photo Booth",
  description: "Kiosk-style photo booth flow in Next.js"
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className="k-shell">
        <BoothProvider>
          <PageTransition>{children}</PageTransition>
        </BoothProvider>
      </body>
    </html>
  );
}
