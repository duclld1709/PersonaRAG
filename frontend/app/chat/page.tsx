import { cookies } from "next/headers";

import ChatLayout from "./ChatLayout";

export default function ChatPage() {
  const cookieStore = cookies();
  const initialCookie = cookieStore.get("sidebar_open");
  const initialOpen = initialCookie?.value !== "0";

  return <ChatLayout initialOpen={initialOpen} />;
}
