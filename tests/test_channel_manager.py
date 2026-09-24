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

def test_unbind_subject(tmp_path: Path):
    bindings_file = tmp_path / "test_bindings.json"
    mgr = ChannelManager(bindings_path=bindings_file)

    mgr.bind_channel(1001, "chem_101", channel_name="#chem-study")
    mgr.bind_channel(1002, "chem_101", channel_name="#chem-help")
    mgr.bind_channel(1003, "math_101", channel_name="#math-general")

    assert len(mgr.get_channels_for_subject("chem_101")) == 2

    # Unbind subject
    unbound_count = mgr.unbind_subject("chem_101")
    assert unbound_count == 2
    assert mgr.get_channels_for_subject("chem_101") == []
    # math_101 should remain
    assert mgr.get_subject_for_channel(1003) == "math_101"

