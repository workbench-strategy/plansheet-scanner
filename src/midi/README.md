# MIDI SysEx Module for Roland JV-1080

This module provides utilities for working with MIDI System Exclusive (SysEx) messages, with a specific focus on the Roland JV-1080 synthesizer.

## Features

- Parse and create SysEx messages
- Handle bulk data dumps and bank dumps
- Send parameter changes to the JV-1080
- Request parameter data from the JV-1080
- Verify checksums for data integrity

## Components

### SysEx Parser (`sysex_parser.py`)

The SysEx Parser provides functions for parsing, validating, and processing SysEx messages. It can:

- Parse single SysEx messages
- Process bulk dumps containing multiple messages
- Handle bank dumps with structured data
- Verify Roland checksums
- Create DT1 (Data Set) and RQ1 (Data Request) messages

### JV-1080 Manager (`jv1080_manager.py`)

The JV-1080 Manager provides a high-level interface for interacting with the Roland JV-1080 synthesizer. It can:

- Load parameter definitions from YAML configuration
- Send parameter changes to the JV-1080
- Request parameter data from the JV-1080
- Request bulk data and bank dumps
- Handle different parameter types (numbers, strings, etc.)

### Configuration (`roland_jv_1080.yaml`)

The YAML configuration file defines:

- Common information (manufacturer ID, model ID, etc.)
- Bank structure for bulk dumps
- Parameter groups and addresses
- Data types and ranges

## Example Usage

### Sending a Parameter

```python
from src.midi import JV1080Manager

# Create a manager
manager = JV1080Manager()

# Send master level parameter
manager.send_parameter(
    group_name="system",
    parameter_name="master_level",
    value=100,  # Value between 0-127
    port_name="JV-1080"  # MIDI port name
)
```

### Requesting a Parameter

```python
from src.midi import JV1080Manager

# Create a manager
manager = JV1080Manager()

# Request master tune parameter
data = manager.request_parameter_data(
    group_name="system",
    parameter_name="master_tune",
    port_name="JV-1080"  # MIDI port name
)

if data:
    # Convert 2-byte value
    tune_value = (data[0] << 4) + data[1]
    print(f"Master tune: {tune_value}")
```

### Requesting Bank Data

```python
from src.midi import JV1080Manager

# Create a manager
manager = JV1080Manager()

# Request patch bank A (bank 0)
bank_data = manager.request_bank_dump(
    bank_type="patch",
    bank_number=0,  # Bank A
    port_name="JV-1080"
)

# Process the bank data
if bank_data and 'patches' in bank_data:
    for patch in bank_data['patches']:
        patch_data = patch['data']
        patch_number = patch['patch_number']
        # Extract patch name from first 12 bytes
        patch_name = ''.join(chr(b) for b in patch_data[0:12]).strip()
        print(f"Patch {patch_number}: {patch_name}")
```

## Demo Script

The `demo_jv1080.py` script provides examples of how to use the SysEx Parser and JV1080 Manager classes. Run it with the following options:

```bash
python demo_jv1080.py --list-ports  # List available MIDI ports
python demo_jv1080.py --port "JV-1080" --send  # Demo sending parameters
python demo_jv1080.py --port "JV-1080" --request  # Demo requesting parameters
python demo_jv1080.py --port "JV-1080" --bulk  # Demo requesting bulk data
python demo_jv1080.py --port "JV-1080" --bank  # Demo requesting bank data
```

## Requirements

- Python 3.6+
- mido library for MIDI I/O
- PyYAML for configuration parsing
