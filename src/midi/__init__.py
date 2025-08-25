"""
MIDI module for SysEx communication with synthesizers.

This package provides classes and utilities for working with MIDI System Exclusive (SysEx)
messages, focusing on Roland JV-1080 support.
"""

from .sysex_parser import SysExParser, SysExMessage
from .jv1080_manager import JV1080Manager

__all__ = ['SysExParser', 'SysExMessage', 'JV1080Manager']
