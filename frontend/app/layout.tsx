import type { Metadata } from "next";
import { Syne, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const syne = Syne({ 
  subsets: ["latin"],
  variable: "--font-syne",
});

const jetbrains = JetBrains_Mono({ 
  subsets: ["latin"],
  variable: "--font-jetbrains-mono",
});

export const metadata: Metadata = {
  title: "NYDRA — Autonomous Data Inspection",
  description: "Drop your data. Get answers. Autonomous agent for data quality.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${syne.variable} ${jetbrains.variable} font-sans antialiased`}>
        {children}
      </body>
    </html>
  );
}
