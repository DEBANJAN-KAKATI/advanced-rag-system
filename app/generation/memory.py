from typing import Dict, List, Any
import time

class ChatMemory:
    def __init__(self, max_history_turns: int = 5):
        self.max_history_turns = max_history_turns
        self._sessions: Dict[str, List[Dict[str, str]]] = {}

    def get_history(self, conversation_id: str) -> List[Dict[str, str]]:
        """Retrieves recent conversation turns for a session."""
        if not conversation_id or conversation_id not in self._sessions:
            return []
        return self._sessions[conversation_id][-self.max_history_turns * 2:]

    def add_turn(self, conversation_id: str, question: str, answer: str):
        """Appends user question and model answer turn."""
        if not conversation_id:
            return
        if conversation_id not in self._sessions:
            self._sessions[conversation_id] = []
            
        self._sessions[conversation_id].append({"role": "user", "content": question})
        self._sessions[conversation_id].append({"role": "assistant", "content": answer})

    def format_history_for_prompt(self, conversation_id: str) -> str:
        """Formats historical context as string for prompt injection."""
        history = self.get_history(conversation_id)
        if not history:
            return ""
        lines = ["--- Prior Conversation History ---"]
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        lines.append("--- End Conversation History ---\n")
        return "\n".join(lines)

    def clear(self, conversation_id: str):
        if conversation_id in self._sessions:
            del self._sessions[conversation_id]

chat_memory = ChatMemory()
