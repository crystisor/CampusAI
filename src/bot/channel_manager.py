import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from src.config import BASE_DIR

logger = logging.getLogger(__name__)

BINDINGS_FILE = BASE_DIR / "storage" / "channel_bindings.json"

class ChannelManager:
    """
    Manages Discord Channel <-> Subject ID bindings.
    Persisted to storage/channel_bindings.json.
    """

    def __init__(self, bindings_path: Optional[Path] = None):
        self.path = bindings_path or BINDINGS_FILE
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._bindings: Dict[str, Dict[str, Any]] = {}
        self.load()

    def load(self):
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._bindings = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load channel bindings: {e}")
                self._bindings = {}
        else:
            self._bindings = {}

    def save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._bindings, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save channel bindings: {e}")

    def bind_channel(self, channel_id: int | str, subject_id: str, channel_name: str = "") -> None:
        """Bind a Discord channel ID to a study subject ID."""
        c_id = str(channel_id)
        self._bindings[c_id] = {
            "channel_id": c_id,
            "subject_id": subject_id.lower().strip(),
            "channel_name": channel_name or f"channel-{c_id}",
        }
        self.save()

    def unbind_channel(self, channel_id: int | str) -> bool:
        """Unbind a Discord channel ID."""
        c_id = str(channel_id)
        if c_id in self._bindings:
            del self._bindings[c_id]
            self.save()
            return True
        return False

    def get_subject_for_channel(self, channel_id: int | str) -> Optional[str]:
        """Returns subject_id bound to channel_id, or None."""
        c_id = str(channel_id)
        record = self._bindings.get(c_id)
        return record["subject_id"] if record else None

    def get_all_bindings(self) -> Dict[str, Dict[str, Any]]:
        """Returns all bindings: {channel_id: {channel_id, subject_id, channel_name}}."""
        return self._bindings

    def get_channels_for_subject(self, subject_id: str) -> list[str]:
        """Returns channel names/ids bound to a subject."""
        sub = subject_id.lower().strip()
        names = []
        for b in self._bindings.values():
            if b.get("subject_id") == sub:
                names.append(b.get("channel_name", b.get("channel_id")))
        return names

    def unbind_subject(self, subject_id: str) -> int:
        """Unbind all Discord channels bound to a subject ID. Returns count of removed bindings."""
        sub = subject_id.lower().strip()
        to_delete = [c_id for c_id, b in self._bindings.items() if b.get("subject_id") == sub]
        for c_id in to_delete:
            del self._bindings[c_id]
        if to_delete:
            self.save()
        return len(to_delete)

# Singleton instance
channel_manager = ChannelManager()
