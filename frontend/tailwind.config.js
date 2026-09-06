/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        nydra: {
          green: "#c8f06e",
          slate: "#1e293b",
          dark: "#0f172a",
        },
      },
      fontFamily: {
        mono: ["var(--font-jetbrains-mono)"],
        sans: ["var(--font-syne)"],
      },
    },
  },
  plugins: [],
};
