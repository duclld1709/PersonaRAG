"use client";

import clsx from "clsx";

import { ConversationSummary } from "../../lib/types";

type Props = {
  conversations: ConversationSummary[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onCreateConversation: () => Promise<void> | void;
  onRename: (id: string) => Promise<void> | void;
};

export function ChatSidebar({
  conversations,
  selectedId,
  onSelect,
  onCreateConversation,
  onRename
}: Props) {
  return (
    <aside className="sidebar">
      <header className="sidebar__header">
        <span className="sidebar__title">Conversations</span>
        <button className="sidebar__action" onClick={() => void onCreateConversation()}>
          + New
        </button>
      </header>

      <ul className="sidebar__list">
        {conversations.length === 0 && <li className="sidebar__empty">No conversations yet.</li>}
        {conversations.map((conversation) => (
          <li key={conversation.id}>
            <div
              className={clsx("sidebar__item", {
                "sidebar__item--active": conversation.id === selectedId
              })}
            >
              <button
                type="button"
                className="sidebar__item-main"
                onClick={() => onSelect(conversation.id)}
              >
                <span className="sidebar__item-title">{conversation.title}</span>
                {conversation.lastMessagePreview && (
                  <span className="sidebar__item-preview">{conversation.lastMessagePreview}</span>
                )}
              </button>
              <button
                type="button"
                className="sidebar__item-rename"
                onClick={(event) => {
                  event.stopPropagation();
                  void onRename(conversation.id);
                }}
              >
                Rename
              </button>
            </div>
          </li>
        ))}
      </ul>
    </aside>
  );
}

export default ChatSidebar;
