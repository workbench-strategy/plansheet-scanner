#!/usr/bin/env python
"""
JV-1080 SysEx Demo

This script demonstrates the use of the SysEx parser and JV1080 Manager classes
to interact with a Roland JV-1080 synthesizer via MIDI System Exclusive messages.
"""

import os
import sys
import logging
import argparse
import time
from typing import List, Dict, Optional, Any
import mido

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the SysEx parser and JV1080 Manager
from src.midi.sysex_parser import SysExParser, SysExMessage
from src.midi.jv1080_manager import JV1080Manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('jv1080_demo')


def list_midi_ports():
    """
    List available MIDI input and output ports
    """
    print("Available MIDI Input Ports:")
    for port in mido.get_input_names():
        print(f"  - {port}")
    
    print("\nAvailable MIDI Output Ports:")
    for port in mido.get_output_names():
        print(f"  - {port}")


def send_parameter_demo(manager: JV1080Manager, port_name: str):
    """
    Demonstrate sending a parameter to the JV-1080
    """
    print("\n--- Sending Parameter Demo ---")
    
    # Send the master level parameter (system group)
    success = manager.send_parameter(
        group_name="system",
        parameter_name="master_level",
        value=100,  # Value between 0-127
        port_name=port_name
    )
    
    if success:
        print("Successfully sent master level parameter")
    else:
        print("Failed to send parameter")
    
    # Send a patch name parameter (string value)
    patch_name = [ord(c) for c in "JV DEMO    "]  # 12 characters, padded with spaces
    success = manager.send_parameter(
        group_name="patch_common",
        parameter_name="patch_name",
        value=patch_name,  # List of ASCII values
        port_name=port_name
    )
    
    if success:
        print("Successfully sent patch name parameter")
    else:
        print("Failed to send parameter")


def request_parameter_demo(manager: JV1080Manager, port_name: str):
    """
    Demonstrate requesting a parameter from the JV-1080
    """
    print("\n--- Requesting Parameter Demo ---")
    
    # Request the master level parameter
    data = manager.request_parameter_data(
        group_name="system",
        parameter_name="master_level",
        port_name=port_name
    )
    
    if data is not None:
        print(f"Received master level value: {data[0]}")
    else:
        print("Failed to receive parameter data")
    
    # Request a patch name
    data = manager.request_parameter_data(
        group_name="patch_common",
        parameter_name="patch_name",
        port_name=port_name
    )
    
    if data is not None:
        # Convert bytes to string
        patch_name = ''.join(chr(b) for b in data).strip()
        print(f"Received patch name: '{patch_name}'")
    else:
        print("Failed to receive patch name")


def request_bulk_data_demo(manager: JV1080Manager, port_name: str):
    """
    Demonstrate requesting bulk data from the JV-1080
    """
    print("\n--- Requesting Bulk Data Demo ---")
    
    # Request system parameters (common settings)
    data = manager.request_bulk_data(
        address_prefix=["00", "00", "00", "00"],
        size=32,  # Request first 32 bytes of system area
        port_name=port_name
    )
    
    if data is not None:
        print(f"Received {len(data)} bytes of system data:")
        print(f"  Master Tune: {data[0:2].hex()}")
        print(f"  Master Level: {data[2]}")
        print(f"  Master Pan: {data[3]}")
    else:
        print("Failed to receive bulk data")


def request_bank_demo(manager: JV1080Manager, port_name: str):
    """
    Demonstrate requesting a bank from the JV-1080
    """
    print("\n--- Requesting Bank Demo ---")
    
    # Request patch bank A (bank number 0)
    bank_data = manager.request_bank_dump(
        bank_type="patch",
        bank_number=0,  # Bank A
        port_name=port_name
    )
    
    if bank_data is not None:
        print(f"Received patch bank data, size: {bank_data['data_size']} bytes")
        
        if 'patches' in bank_data:
            print(f"Number of patches: {len(bank_data['patches'])}")
            
            # Print the first patch name
            if len(bank_data['patches']) > 0:
                patch_data = bank_data['patches'][0]['data']
                patch_name_bytes = patch_data[0:12]
                patch_name = ''.join(chr(b) for b in patch_name_bytes).strip()
                print(f"First patch name: '{patch_name}'")
    else:
        print("Failed to receive bank data")


def main():
    """
    Main function to run the demo
    """
    parser = argparse.ArgumentParser(description='JV-1080 SysEx Demo')
    
    parser.add_argument(
        '--list-ports',
        action='store_true',
        help='List available MIDI ports and exit'
    )
    
    parser.add_argument(
        '--port',
        type=str,
        help='MIDI port name to use'
    )
    
    parser.add_argument(
        '--send',
        action='store_true',
        help='Demonstrate sending parameters'
    )
    
    parser.add_argument(
        '--request',
        action='store_true',
        help='Demonstrate requesting parameters'
    )
    
    parser.add_argument(
        '--bulk',
        action='store_true',
        help='Demonstrate requesting bulk data'
    )
    
    parser.add_argument(
        '--bank',
        action='store_true',
        help='Demonstrate requesting bank data'
    )
    
    args = parser.parse_args()
    
    # Just list MIDI ports if requested
    if args.list_ports:
        list_midi_ports()
        return 0
    
    # Create JV1080Manager
    manager = JV1080Manager()
    
    if args.port is None:
        print("Error: No MIDI port specified. Use --port or --list-ports to see available ports.")
        return 1
    
    # Run requested demos
    if args.send:
        send_parameter_demo(manager, args.port)
    
    if args.request:
        request_parameter_demo(manager, args.port)
    
    if args.bulk:
        request_bulk_data_demo(manager, args.port)
    
    if args.bank:
        request_bank_demo(manager, args.port)
    
    # If no specific demo was requested, run them all
    if not (args.send or args.request or args.bulk or args.bank):
        send_parameter_demo(manager, args.port)
        time.sleep(1)  # Give the device time to process
        
        request_parameter_demo(manager, args.port)
        time.sleep(1)
        
        request_bulk_data_demo(manager, args.port)
        time.sleep(1)
        
        request_bank_demo(manager, args.port)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
