import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-sans",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Pixel_Info - AI Image Caption Generator",
  description: "Generate high-quality captions for your images using deep learning (CNN-LSTM and Beam Search).",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${inter.variable} h-full antialiased`}>
      <body className="min-h-full flex flex-col bg-[#0B0F19] text-[#EDF2F7]">
        {children}
      </body>
    </html>
  );
}
