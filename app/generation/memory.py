import re
import threading
from collections import OrderedDict
from typing import Dict, List
from app.core.config import settings

# Assistant answers are stored truncated; full answers would bloat every later prompt.
MAX_STORED_ANSWER_CHARS = 1200


class ChatMemory:
    """In-memory conversation history, bounded per conversation and in number of conversations (LRU)."""

    def __init__(self, max_history_turns: int = None, max_conversations: int = None):
        self.max_history_turns = max_history_turns or settings.MEMORY_TURNS
        self.max_conversations = max_conversations or settings.MAX_CONVERSATIONS
        self._sessions: "OrderedDict[str, List[Dict[str, str]]]" = OrderedDict()
        self._lock = threading.Lock()

    def get_history(self, conversation_id: str) -> List[Dict[str, str]]:
        with self._lock:
            if not conversation_id or conversation_id not in self._sessions:
                return []
            self._sessions.move_to_end(conversation_id)
            return list(self._sessions[conversation_id][-self.max_history_turns * 2:])

    def add_turn(self, conversation_id: str, question: str, answer: str):
        if not conversation_id:
            return
        answer = re.sub(r"[ \t]*(?:\[\d{1,3}\])+", "", answer).strip()  # citation markers mean nothing out of context
        if len(answer) > MAX_STORED_ANSWER_CHARS:
            answer = answer[:MAX_STORED_ANSWER_CHARS].rsplit(" ", 1)[0] + " ..."
        with self._lock:
            turns = self._sessions.setdefault(conversation_id, [])
            turns.append({"role": "user", "content": question})
            turns.append({"role": "assistant", "content": answer})
            del turns[:-self.max_history_turns * 2]
            self._sessions.move_to_end(conversation_id)
            while len(self._sessions) > self.max_conversations:
                self._sessions.popitem(last=False)

    def format_history_for_prompt(self, conversation_id: str) -> str:
        history = self.get_history(conversation_id)
        if not history:
            return ""
        lines = ["Conversation so far:"]
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines) + "\n\n"

    def clear(self, conversation_id: str):
        with self._lock:
            self._sessions.pop(conversation_id, None)


chat_memory = ChatMemory()
