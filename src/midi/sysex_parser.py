#!/usr/bin/env python
"""
SysEx Parser Module

This module provides utilities for parsing and processing MIDI System Exclusive (SysEx)
messages, with a focus on Roland JV-1080 data.

Features:
- Parse SysEx messages to extract headers, address, data, and checksums
- Handle bulk data dumps and bank dumps
- Verify checksums for data integrity
- Convert between different data formats (bytes, hex strings, etc.)
"""

from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import struct
import re
from dataclasses import dataclass

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class SysExMessage:
    """
    Represents a parsed SysEx message with all its components.
    
    Attributes:
        raw_data: The complete raw SysEx message as bytes
        manufacturer_id: The manufacturer ID byte(s)
        device_id: The device ID byte
        model_id: The model ID byte
        command_id: The command ID byte (e.g., DT1, RQ1)
        address: The address bytes (usually 4 bytes for Roland)
        data: The data bytes (may be empty for RQ1 messages)
        checksum: The checksum byte
        size: For RQ1 messages, the size bytes requested
    """
    raw_data: bytes
    manufacturer_id: bytes
    device_id: bytes
    model_id: bytes 
    command_id: bytes
    address: bytes
    data: bytes
    checksum: int
    size: Optional[bytes] = None
    
    @property
    def is_dt1(self) -> bool:
        """Returns True if this is a Data Set 1 (DT1) message"""
        # Roland DT1 command is typically 0x12
        return self.command_id == b'\x12'
    
    @property
    def is_rq1(self) -> bool:
        """Returns True if this is a Data Request 1 (RQ1) message"""
        # Roland RQ1 command is typically 0x11
        return self.command_id == b'\x11'
    
    @property
    def address_hex(self) -> str:
        """Returns the address as a hex string"""
        return ' '.join(f"{b:02X}" for b in self.address)
    
    @property
    def data_hex(self) -> str:
        """Returns the data as a hex string"""
        return ' '.join(f"{b:02X}" for b in self.data)
    
    @property
    def size_hex(self) -> Optional[str]:
        """Returns the size bytes as a hex string (for RQ1 messages)"""
        if self.size:
            return ' '.join(f"{b:02X}" for b in self.size)
        return None

    def __str__(self) -> str:
        """String representation of the SysEx message"""
        msg_type = "DT1" if self.is_dt1 else ("RQ1" if self.is_rq1 else "Unknown")
        result = f"SysEx {msg_type} Message:\n"
        result += f"  Manufacturer ID: {' '.join(f'{b:02X}' for b in self.manufacturer_id)}\n"
        result += f"  Device ID: {self.device_id.hex().upper()}\n"
        result += f"  Model ID: {self.model_id.hex().upper()}\n"
        result += f"  Command ID: {self.command_id.hex().upper()}\n"
        result += f"  Address: {self.address_hex}\n"
        
        if self.is_dt1:
            result += f"  Data: {self.data_hex}\n"
        elif self.is_rq1:
            result += f"  Size: {self.size_hex}\n"
            
        result += f"  Checksum: {self.checksum:02X}\n"
        return result


class SysExParser:
    """
    Parser for MIDI System Exclusive messages, with focus on Roland format.
    
    This class provides methods to parse, validate, and process SysEx messages,
    including bulk and bank dumps.
    """
    
    # Constants
    SYSEX_START = 0xF0
    SYSEX_END = 0xF7
    ROLAND_MANUFACTURER_ID = 0x41
    
    def __init__(self):
        """Initialize the SysEx parser"""
        pass
        
    @staticmethod
    def is_valid_sysex(data: bytes) -> bool:
        """
        Check if the given data represents a valid SysEx message.
        
        Args:
            data: The SysEx data as bytes
            
        Returns:
            bool: True if the data is a valid SysEx message
        """
        return (len(data) >= 4 and
                data[0] == SysExParser.SYSEX_START and
                data[-1] == SysExParser.SYSEX_END)
    
    @staticmethod
    def calculate_roland_checksum(address_and_data: bytes) -> int:
        """
        Calculate the Roland-style checksum for SysEx messages.
        
        The Roland checksum is calculated by taking the sum of all address and data bytes,
        then taking the lower 7 bits of the 2's complement of that sum.
        
        Args:
            address_and_data: The address and data bytes to calculate the checksum for
            
        Returns:
            int: The calculated checksum value
        """
        checksum = 0
        for byte in address_and_data:
            checksum = (checksum + byte) & 0xFF
        
        # Return 2's complement of lower 7 bits
        return (128 - (checksum & 0x7F)) & 0x7F
    
    @staticmethod
    def verify_checksum(message: bytes) -> bool:
        """
        Verify the checksum of a Roland SysEx message.
        
        Args:
            message: The SysEx message including checksum
            
        Returns:
            bool: True if the checksum is valid
        """
        # For Roland messages, the checksum is calculated over address and data bytes
        # Extract address and data bytes, excluding:
        # F0, manufacturer ID, device ID, model ID, command ID, and F7
        start_idx = 5  # Skip F0, manufacturer ID, device ID, model ID, command ID
        end_idx = -2   # Skip checksum and F7
        
        address_and_data = message[start_idx:end_idx]
        calculated_checksum = SysExParser.calculate_roland_checksum(address_and_data)
        received_checksum = message[-2]
        
        return calculated_checksum == received_checksum
    
    @staticmethod
    def hex_string_to_bytes(hex_string: str) -> bytes:
        """
        Convert a string of hex values to bytes.
        
        Args:
            hex_string: String of hex values, can include spaces
            
        Returns:
            bytes: The converted bytes
        """
        # Remove spaces and non-hex characters
        clean_hex = re.sub(r'[^0-9a-fA-F]', '', hex_string)
        # Convert to bytes
        return bytes.fromhex(clean_hex)
    
    @staticmethod
    def parse_message(data: bytes) -> Optional[SysExMessage]:
        """
        Parse a SysEx message and return its components.
        
        Args:
            data: The raw SysEx message data
            
        Returns:
            Optional[SysExMessage]: A parsed SysEx message or None if parsing fails
        """
        if not SysExParser.is_valid_sysex(data):
            logger.warning("Invalid SysEx message format")
            return None
            
        try:
            # Basic format validation
            if len(data) < 10:  # Minimum length for a valid Roland SysEx
                logger.warning("SysEx message too short")
                return None
                
            # Extract components - Roland format
            # F0 41 10 6A 12 [address] [data] [checksum] F7
            manufacturer_id = bytes([data[1]])
            device_id = bytes([data[2]])
            model_id = bytes([data[3]])
            command_id = bytes([data[4]])
            
            # Handle both DT1 and RQ1 messages
            if command_id == b'\x12':  # DT1 message
                # Standard format has 4 address bytes
                address = data[5:9]
                # Data is between address and checksum
                data_bytes = data[9:-2]
                checksum = data[-2]
                
                return SysExMessage(
                    raw_data=data,
                    manufacturer_id=manufacturer_id,
                    device_id=device_id,
                    model_id=model_id,
                    command_id=command_id,
                    address=address,
                    data=data_bytes,
                    checksum=checksum
                )
                
            elif command_id == b'\x11':  # RQ1 message
                # Standard format has 4 address bytes and 4 size bytes
                address = data[5:9]
                size = data[9:13]
                checksum = data[-2]
                
                return SysExMessage(
                    raw_data=data,
                    manufacturer_id=manufacturer_id,
                    device_id=device_id,
                    model_id=model_id,
                    command_id=command_id,
                    address=address,
                    data=b'',  # No data in RQ1 messages
                    checksum=checksum,
                    size=size
                )
                
            else:
                logger.warning(f"Unsupported command ID: {command_id.hex()}")
                return None
                
        except Exception as e:
            logger.error(f"Error parsing SysEx message: {e}")
            return None
    
    def parse_bulk_dump(self, data: bytes) -> List[SysExMessage]:
        """
        Parse a bulk dump containing multiple SysEx messages.
        
        Args:
            data: The raw data potentially containing multiple SysEx messages
            
        Returns:
            List[SysExMessage]: List of parsed SysEx messages
        """
        messages = []
        pos = 0
        
        while pos < len(data):
            # Find the start of a SysEx message
            while pos < len(data) and data[pos] != self.SYSEX_START:
                pos += 1
                
            if pos >= len(data):
                break
                
            # Find the end of this SysEx message
            start = pos
            pos += 1
            while pos < len(data) and data[pos] != self.SYSEX_END:
                pos += 1
                
            if pos < len(data) and data[pos] == self.SYSEX_END:
                # Extract the complete message including START and END bytes
                message_data = data[start:pos+1]
                parsed_message = self.parse_message(message_data)
                
                if parsed_message:
                    messages.append(parsed_message)
                    
                pos += 1
            else:
                # No END byte found, break the loop
                break
                
        return messages
    
    def parse_bank_dump(self, data: bytes) -> Dict[str, Any]:
        """
        Parse a bank dump containing multiple related SysEx messages.
        
        This method specifically handles bank dumps which contain multiple patches
        or settings that should be considered as a cohesive unit.
        
        Args:
            data: The raw data containing a complete bank dump
            
        Returns:
            Dict[str, Any]: Structured representation of the bank data
        """
        # Parse all messages in the dump
        messages = self.parse_bulk_dump(data)
        
        # Group messages by their address prefix to identify banks
        banks = {}
        for msg in messages:
            if not msg.is_dt1:
                continue
                
            # Extract bank info from address (implementation depends on specific device)
            addr_prefix = msg.address[:2].hex().upper()
            
            if addr_prefix not in banks:
                banks[addr_prefix] = []
                
            banks[addr_prefix].append(msg)
            
        # Structure the result
        result = {
            "messages_count": len(messages),
            "banks": {},
            "raw_messages": messages
        }
        
        # Process each bank group
        for addr_prefix, bank_messages in banks.items():
            # Sort messages by address for consistency
            sorted_messages = sorted(bank_messages, key=lambda m: m.address)
            
            # Extract all data into a continuous block for this bank
            bank_data = b''.join(msg.data for msg in sorted_messages)
            
            result["banks"][addr_prefix] = {
                "message_count": len(bank_messages),
                "data_size": len(bank_data),
                "data": bank_data,
                "addresses": [msg.address_hex for msg in sorted_messages]
            }
            
        return result
    
    def create_dt1_message(self, manufacturer_id: bytes, device_id: bytes,
                          model_id: bytes, address: bytes, data: bytes) -> bytes:
        """
        Create a DT1 (Data Set 1) SysEx message.
        
        Args:
            manufacturer_id: Manufacturer ID byte(s)
            device_id: Device ID byte
            model_id: Model ID byte
            address: Address bytes (usually 4 bytes)
            data: Data bytes
            
        Returns:
            bytes: The complete DT1 SysEx message
        """
        # Combine address and data for checksum calculation
        address_and_data = address + data
        checksum = self.calculate_roland_checksum(address_and_data)
        
        # Assemble the complete message
        message = (
            bytes([self.SYSEX_START]) +
            manufacturer_id +
            device_id +
            model_id +
            bytes([0x12]) +  # DT1 command ID
            address +
            data +
            bytes([checksum]) +
            bytes([self.SYSEX_END])
        )
        
        return message
    
    def create_rq1_message(self, manufacturer_id: bytes, device_id: bytes,
                          model_id: bytes, address: bytes, size: bytes) -> bytes:
        """
        Create an RQ1 (Data Request 1) SysEx message.
        
        Args:
            manufacturer_id: Manufacturer ID byte(s)
            device_id: Device ID byte
            model_id: Model ID byte
            address: Address bytes (usually 4 bytes)
            size: Size bytes (usually 4 bytes) specifying how much data to request
            
        Returns:
            bytes: The complete RQ1 SysEx message
        """
        # Combine address and size for checksum calculation
        address_and_size = address + size
        checksum = self.calculate_roland_checksum(address_and_size)
        
        # Assemble the complete message
        message = (
            bytes([self.SYSEX_START]) +
            manufacturer_id +
            device_id +
            model_id +
            bytes([0x11]) +  # RQ1 command ID
            address +
            size +
            bytes([checksum]) +
            bytes([self.SYSEX_END])
        )
        
        return message
