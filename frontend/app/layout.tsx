import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RAG-MCwiki | Minecraft 智能助手",
  description: "基于本地大模型的 Minecraft Wiki 知识库检索增强生成系统",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
