#!/usr/bin/env python
"""
JV-1080 Manager Module

This module provides a high-level interface for interacting with the Roland JV-1080
synthesizer via MIDI System Exclusive (SysEx) messages. It supports both sending
parameters (DT1) and requesting data (RQ1).

Features:
- Load parameter definitions from YAML configuration
- Send parameter changes to the JV-1080
- Request parameter data from the JV-1080
- Support for bank and bulk dumps
"""

import os
import time
import logging
import yaml
import mido
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

from .sysex_parser import SysExParser, SysExMessage

# Set up logging
logger = logging.getLogger(__name__)


class JV1080Manager:
    """
    Manager for Roland JV-1080 synthesizer.
    
    This class provides methods to interact with a Roland JV-1080 synthesizer
    via MIDI SysEx messages, supporting both sending and requesting parameters.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the JV-1080 Manager.
        
        Args:
            config_path: Path to the YAML configuration file containing parameter definitions.
                         If None, will look for 'roland_jv_1080.yaml' in the same directory.
        """
        self.parser = SysExParser()
        self.config = {}
        
        # Load configuration
        if config_path is None:
            # Look for the config file in the same directory as this script
            config_path = os.path.join(os.path.dirname(__file__), "roland_jv_1080.yaml")
        
        self._load_config(config_path)
        
        # Cache for received SysEx messages
        self.received_messages = []
        
    def _load_config(self, config_path: str) -> None:
        """
        Load parameter definitions from a YAML configuration file.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Loaded JV-1080 configuration from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            # Initialize with empty config
            self.config = {"common_info": {}, "parameter_groups": {}}
    
    def _get_parameter_address(self, group_name: str, parameter_name: str) -> Optional[List[str]]:
        """
        Get the address of a specific parameter from the configuration.
        
        Args:
            group_name: The parameter group (e.g., 'patch', 'performance')
            parameter_name: The specific parameter name
            
        Returns:
            Optional[List[str]]: List of hex strings representing the parameter address or None if not found
        """
        try:
            # Look up the parameter group
            group = self.config["parameter_groups"].get(group_name)
            if group is None:
                logger.error(f"Parameter group '{group_name}' not found in configuration")
                return None
                
            # Look up the parameter
            param = group["parameters"].get(parameter_name)
            if param is None:
                logger.error(f"Parameter '{parameter_name}' not found in group '{group_name}'")
                return None
                
            # Get the address
            if "address_hex" not in param:
                logger.error(f"No address defined for parameter '{parameter_name}' in group '{group_name}'")
                return None
                
            return param["address_hex"]
            
        except Exception as e:
            logger.error(f"Error getting parameter address: {e}")
            return None
    
    def _get_parameter_size(self, group_name: str, parameter_name: str) -> Optional[List[str]]:
        """
        Get the size of a specific parameter from the configuration.
        
        Args:
            group_name: The parameter group (e.g., 'patch', 'performance')
            parameter_name: The specific parameter name
            
        Returns:
            Optional[List[str]]: List of hex strings representing the parameter size or None if not found
        """
        try:
            # Look up the parameter group
            group = self.config["parameter_groups"].get(group_name)
            if group is None:
                logger.error(f"Parameter group '{group_name}' not found in configuration")
                return None
                
            # Look up the parameter
            param = group["parameters"].get(parameter_name)
            if param is None:
                logger.error(f"Parameter '{parameter_name}' not found in group '{group_name}'")
                return None
                
            # Get the size
            if "size_hex" in param:
                return param["size_hex"]
                
            # If size isn't explicitly defined, determine it from the data type
            if "data_type" in param:
                data_type = param["data_type"]
                if data_type == "byte":
                    return ["00", "00", "00", "01"]
                elif data_type == "word":
                    return ["00", "00", "00", "02"]
                # Add more data types as needed
                
            # Default to 1 byte if we can't determine size
            logger.warning(f"Could not determine size for parameter '{parameter_name}', defaulting to 1 byte")
            return ["00", "00", "00", "01"]
            
        except Exception as e:
            logger.error(f"Error getting parameter size: {e}")
            return ["00", "00", "00", "01"]  # Default to 1 byte
    
    def _hex_list_to_bytes(self, hex_list: List[str]) -> bytes:
        """
        Convert a list of hex strings to bytes.
        
        Args:
            hex_list: List of hex strings (e.g., ["00", "7F", "40"])
            
        Returns:
            bytes: The converted bytes
        """
        return bytes([int(h, 16) for h in hex_list])
    
    def _get_common_info(self, info_key: str) -> Optional[str]:
        """
        Get common information from the configuration.
        
        Args:
            info_key: The key for the information to retrieve
            
        Returns:
            Optional[str]: The requested information or None if not found
        """
        return self.config.get("common_info", {}).get(info_key)
    
    def _get_midi_port(self, port_name: str) -> Optional[mido.ports.BasePort]:
        """
        Get a MIDI output port by name.
        
        Args:
            port_name: The name of the MIDI port to use
            
        Returns:
            Optional[mido.ports.BasePort]: The MIDI output port or None if not found
        """
        try:
            # Check if the port exists
            available_ports = mido.get_output_names()
            if port_name not in available_ports:
                logger.error(f"MIDI port '{port_name}' not found. Available ports: {available_ports}")
                return None
                
            # Open the port
            return mido.open_output(port_name)
            
        except Exception as e:
            logger.error(f"Error opening MIDI port '{port_name}': {e}")
            return None
            
    def _setup_callback(self, port_name: str) -> Optional[mido.ports.BasePort]:
        """
        Set up a callback for receiving SysEx messages.
        
        Args:
            port_name: The name of the MIDI input port to use
            
        Returns:
            Optional[mido.ports.BasePort]: The MIDI input port or None if not found
        """
        try:
            # Check if the port exists
            available_ports = mido.get_input_names()
            if port_name not in available_ports:
                logger.error(f"MIDI port '{port_name}' not found. Available ports: {available_ports}")
                return None
                
            # Open the port
            input_port = mido.open_input(port_name)
            
            # Set up callback
            def callback(message):
                if message.type == 'sysex':
                    self.received_messages.append(message.data)
                    
            input_port.callback = callback
            return input_port
            
        except Exception as e:
            logger.error(f"Error setting up callback for MIDI port '{port_name}': {e}")
            return None
    
    def send_parameter(self, group_name: str, parameter_name: str,
                     value: Union[int, List[int]], port_name: str,
                     device_id_hex: str = "10") -> bool:
        """
        Send a parameter to the JV-1080 using DT1 message.
        
        Args:
            group_name: The parameter group name
            parameter_name: The parameter name
            value: The value to send (int or list of ints)
            port_name: The MIDI port name to use
            device_id_hex: The device ID in hex (default: "10" for unit #17)
            
        Returns:
            bool: True if the parameter was sent successfully
        """
        # Get the parameter address
        address_hex = self._get_parameter_address(group_name, parameter_name)
        if not address_hex:
            return False
            
        # Convert the value to a list if it's a single int
        if isinstance(value, int):
            value = [value]
            
        # Convert the address to bytes
        address_bytes = self._hex_list_to_bytes(address_hex)
        
        # Convert the value to bytes
        data_bytes = bytes(value)
        
        # Get common information
        manufacturer_id_hex = self._get_common_info("manufacturer_id_hex")
        model_id_hex = self._get_common_info("model_id_hex")
        
        if not manufacturer_id_hex or not model_id_hex:
            logger.error("Missing common information in configuration")
            return False
            
        # Convert to bytes
        manufacturer_id = bytes([int(manufacturer_id_hex, 16)])
        device_id = bytes([int(device_id_hex, 16)])
        model_id = bytes([int(model_id_hex, 16)])
        
        # Create DT1 message
        message = self.parser.create_dt1_message(
            manufacturer_id=manufacturer_id,
            device_id=device_id,
            model_id=model_id,
            address=address_bytes,
            data=data_bytes
        )
        
        # Send the message
        try:
            port = self._get_midi_port(port_name)
            if port is None:
                return False
                
            # Create MIDI message
            midi_msg = mido.Message('sysex', data=message[1:-1])  # Remove F0 and F7
            port.send(midi_msg)
            port.close()
            
            logger.info(f"Sent parameter '{parameter_name}' in group '{group_name}' to JV-1080")
            return True
            
        except Exception as e:
            logger.error(f"Error sending parameter: {e}")
            return False
    
    def request_parameter_data(self, group_name: str, parameter_name: str, 
                             port_name: str, device_id_hex: str = "10") -> Optional[bytes]:
        """
        Request parameter data from the JV-1080 using RQ1 message.
        
        Args:
            group_name: The parameter group name
            parameter_name: The parameter name
            port_name: The MIDI port name to use
            device_id_hex: The device ID in hex (default: "10" for unit #17)
            
        Returns:
            Optional[bytes]: The received data or None if request failed
        """
        # Get the parameter address and size
        address_hex = self._get_parameter_address(group_name, parameter_name)
        size_hex = self._get_parameter_size(group_name, parameter_name)
        
        if not address_hex or not size_hex:
            return None
            
        # Request data using the generic method
        return self.request_data(
            address_hex=address_hex,
            size_hex=size_hex,
            port_name=port_name,
            device_id_hex=device_id_hex
        )
    
    def request_data(self, address_hex: List[str], size_hex: List[str], 
                   port_name: str, device_id_hex: str = "10") -> Optional[bytes]:
        """
        Request data from the JV-1080 using RQ1 message.
        
        Args:
            address_hex: List of hex strings representing the address
            size_hex: List of hex strings representing the size
            port_name: The MIDI port name to use
            device_id_hex: The device ID in hex (default: "10" for unit #17)
            
        Returns:
            Optional[bytes]: The received data or None if request failed
        """
        # Convert address and size to bytes
        address_bytes = self._hex_list_to_bytes(address_hex)
        size_bytes = self._hex_list_to_bytes(size_hex)
        
        # Get common information
        manufacturer_id_hex = self._get_common_info("manufacturer_id_hex")
        model_id_hex = self._get_common_info("model_id_hex")
        command_id_rq1_hex = self._get_common_info("command_id_rq1_hex")
        
        if not manufacturer_id_hex or not model_id_hex or not command_id_rq1_hex:
            logger.error("Missing common information in configuration")
            return None
            
        # Convert to bytes
        manufacturer_id = bytes([int(manufacturer_id_hex, 16)])
        device_id = bytes([int(device_id_hex, 16)])
        model_id = bytes([int(model_id_hex, 16)])
        
        # Create RQ1 message
        message = self.parser.create_rq1_message(
            manufacturer_id=manufacturer_id,
            device_id=device_id,
            model_id=model_id,
            address=address_bytes,
            size=size_bytes
        )
        
        # Clear previously received messages
        self.received_messages = []
        
        # Set up callback for receiving response
        input_port = self._setup_callback(port_name)
        if input_port is None:
            return None
            
        # Send the request
        try:
            port = self._get_midi_port(port_name)
            if port is None:
                input_port.close()
                return None
                
            # Create MIDI message
            midi_msg = mido.Message('sysex', data=message[1:-1])  # Remove F0 and F7
            port.send(midi_msg)
            
            # Wait for response (with timeout)
            start_time = time.time()
            timeout = 2.0  # 2 seconds timeout
            
            while not self.received_messages and time.time() - start_time < timeout:
                time.sleep(0.01)
                
            port.close()
            input_port.close()
            
            # Process response
            if not self.received_messages:
                logger.warning("No response received from JV-1080")
                return None
                
            # Parse the response (should be a DT1 message)
            response = self.received_messages[0]
            parsed_message = self.parser.parse_message(bytes([0xF0]) + response + bytes([0xF7]))
            
            if not parsed_message or not parsed_message.is_dt1:
                logger.error("Invalid response received")
                return None
                
            logger.info(f"Received data from JV-1080: {len(parsed_message.data)} bytes")
            return parsed_message.data
            
        except Exception as e:
            logger.error(f"Error requesting data: {e}")
            if input_port:
                input_port.close()
            return None
            
    def request_bulk_data(self, address_prefix: List[str], size: int, 
                        port_name: str, device_id_hex: str = "10") -> Optional[bytes]:
        """
        Request a bulk data dump from the JV-1080.
        
        This is useful for requesting a complete bank or multiple parameters at once.
        
        Args:
            address_prefix: List of hex strings representing the address prefix
            size: Size of data to request in bytes
            port_name: The MIDI port name to use
            device_id_hex: The device ID in hex (default: "10" for unit #17)
            
        Returns:
            Optional[bytes]: The received data or None if request failed
        """
        # Ensure we have a complete 4-byte address
        if len(address_prefix) < 4:
            address_hex = address_prefix + ["00"] * (4 - len(address_prefix))
        else:
            address_hex = address_prefix[:4]
            
        # Convert size to 4-byte hex list
        size_hex = [
            format((size >> 24) & 0xFF, '02X'),
            format((size >> 16) & 0xFF, '02X'),
            format((size >> 8) & 0xFF, '02X'),
            format(size & 0xFF, '02X')
        ]
        
        # Request the data
        return self.request_data(
            address_hex=address_hex,
            size_hex=size_hex,
            port_name=port_name,
            device_id_hex=device_id_hex
        )
        
    def request_bank_dump(self, bank_type: str, bank_number: int,
                        port_name: str, device_id_hex: str = "10") -> Optional[Dict[str, Any]]:
        """
        Request a complete bank dump from the JV-1080.
        
        Args:
            bank_type: The type of bank ('patch', 'performance', etc.)
            bank_number: The bank number
            port_name: The MIDI port name to use
            device_id_hex: The device ID in hex (default: "10" for unit #17)
            
        Returns:
            Optional[Dict[str, Any]]: Dictionary containing the bank data or None if failed
        """
        # Get bank information from config
        bank_info = self.config.get("banks", {}).get(bank_type)
        if not bank_info:
            logger.error(f"Bank type '{bank_type}' not found in configuration")
            return None
            
        # Get address and size
        if "address_format" not in bank_info or "size" not in bank_info:
            logger.error(f"Missing address_format or size for bank type '{bank_type}'")
            return None
            
        # Calculate address using bank number
        try:
            address_hex = []
            for byte_format in bank_info["address_format"]:
                if isinstance(byte_format, str):
                    # Fixed value
                    address_hex.append(byte_format)
                elif isinstance(byte_format, dict) and "bank_offset" in byte_format:
                    # Value based on bank number
                    offset = int(byte_format["bank_offset"], 16)
                    value = (bank_number * offset) & 0xFF
                    address_hex.append(format(value, '02X'))
                else:
                    # Default to 00
                    address_hex.append("00")
                    
            # Get size in bytes
            size = int(bank_info["size"], 16)
            
            # Request the data
            data = self.request_bulk_data(
                address_prefix=address_hex,
                size=size,
                port_name=port_name,
                device_id_hex=device_id_hex
            )
            
            if data is None:
                return None
                
            # Process the data based on bank type
            # This will depend on the specific format of each bank type
            result = {
                "bank_type": bank_type,
                "bank_number": bank_number,
                "data_size": len(data),
                "raw_data": data,
                # Add more structured data based on bank type
            }
            
            # Example: for patch banks, we might parse individual patches
            if bank_type == "patch":
                patches = []
                patch_size = bank_info.get("patch_size", 128)  # Default patch size
                
                for i in range(0, len(data), patch_size):
                    if i + patch_size <= len(data):
                        patch_data = data[i:i+patch_size]
                        patch_number = i // patch_size
                        patches.append({
                            "patch_number": patch_number,
                            "data": patch_data
                            # Add more patch-specific information
                        })
                        
                result["patches"] = patches
                
            return result
            
        except Exception as e:
            logger.error(f"Error requesting bank dump: {e}")
            return None
