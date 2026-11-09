from typing import Dict, List, Any
import time


class SessionMemory:
    """
    In-memory conversation manager that maintains chat history
    per session_id with automatic sliding window truncation.
    """

    def __init__(self, max_history_turns: int = 10, ttl_seconds: int = 3600):
        self.sessions: Dict[str, List[Dict[str, Any]]] = {}
        self.last_accessed: Dict[str, float] = {}
        self.max_history_turns = max_history_turns
        self.ttl_seconds = ttl_seconds

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieves history for a session if not expired."""
        now = time.time()
        if session_id in self.last_accessed:
            if now - self.last_accessed[session_id] > self.ttl_seconds:
                # Expired session
                self.sessions.pop(session_id, None)
                self.last_accessed.pop(session_id, None)
                return []

        self.last_accessed[session_id] = now
        return self.sessions.get(session_id, [])

    def append_message(self, session_id: str, message: Dict[str, Any]) -> None:
        """Appends a message to the session history while respecting the sliding window."""
        if session_id not in self.sessions:
            self.sessions[session_id] = []

        self.sessions[session_id].append(message)
        self.last_accessed[session_id] = time.time()

        # Sliding window: keep only the most recent (max_history_turns * 2) messages
        max_messages = self.max_history_turns * 2
        if len(self.sessions[session_id]) > max_messages:
            self.sessions[session_id] = self.sessions[session_id][-max_messages:]

    def clear(self, session_id: str) -> None:
        """Clears memory for a specific session."""
        self.sessions.pop(session_id, None)
        self.last_accessed.pop(session_id, None)


# Global in-memory session manager
memory_manager = SessionMemory(max_history_turns=6, ttl_seconds=3600)

