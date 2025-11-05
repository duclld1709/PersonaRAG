import "./globals.css";

import { ReactNode } from "react";

export const metadata = {
  title: "RAG Chatbot",
  description: "Conversational interface powered by a retrieval-augmented backend."
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="app-root">{children}</body>
    </html>
  );
}
