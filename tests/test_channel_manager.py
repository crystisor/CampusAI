import pytest
from pathlib import Path
from src.bot.channel_manager import ChannelManager

def test_channel_manager(tmp_path: Path):
    bindings_file = tmp_path / "test_bindings.json"
    mgr = ChannelManager(bindings_path=bindings_file)

    mgr.bind_channel(123456, "physics_101", channel_name="#physics-chat")
    assert mgr.get_subject_for_channel(123456) == "physics_101"
    assert mgr.get_subject_for_channel("123456") == "physics_101"

    channels = mgr.get_channels_for_subject("physics_101")
    assert "#physics-chat" in channels

    # Unbind
    removed = mgr.unbind_channel(123456)
    assert removed is True
    assert mgr.get_subject_for_channel(123456) is None
