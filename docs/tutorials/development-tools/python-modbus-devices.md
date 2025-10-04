# Industrial Data Harvesting: Python Modbus Device Communication

**Objective**: Master industrial device communication using Modbus protocol to connect to PLCs, sensors, and automation equipment. When you need to integrate with industrial systems, when you want to monitor manufacturing equipment, when you're building IoT solutions for industrial environments—Python Modbus becomes your weapon of choice.

Industrial devices speak a different language. Let's learn to communicate with PLCs, HMIs, and sensors using the universal language of industrial automation: Modbus protocol.

## 0) Prerequisites (Read Once, Live by Them)

### The Five Commandments

1. **Understand Modbus protocol**
   - Modbus RTU over serial communication
   - Modbus TCP over Ethernet networks
   - Function codes and data types
   - Register addressing and data mapping

2. **Master Python Modbus libraries**
   - pymodbus for client/server operations
   - pyserial for serial communication
   - asyncio for concurrent operations
   - Error handling and connection management

3. **Know your industrial devices**
   - PLCs and programmable controllers
   - HMIs and operator interfaces
   - Sensors and measurement devices
   - SCADA systems and data acquisition

4. **Validate everything**
   - Test device connectivity and communication
   - Verify data reading and writing operations
   - Check error handling and recovery
   - Monitor performance and reliability

5. **Plan for production**
   - Design for industrial environments
   - Enable robust error handling and recovery
   - Support multiple device types and protocols
   - Document operational procedures

**Why These Principles**: Industrial communication requires understanding both protocol specifications and Python implementation. Understanding these patterns prevents communication failures and enables reliable industrial automation.

## 1) Setup and Dependencies

### Required Packages

```bash
# Install required packages
pip install pymodbus pyserial asyncio-mqtt paho-mqtt

# For advanced features
pip install pymodbus[serial] pymodbus[asyncio] pymodbus[twisted]

# For data visualization
pip install matplotlib pandas numpy

# For web interfaces
pip install flask fastapi uvicorn
```

**Why Package Setup Matters**: Proper dependencies enable reliable Modbus communication. Understanding these patterns prevents installation issues and enables professional industrial automation.

### Project Structure

```
modbus-iot/
├── devices/
│   ├── plc_controller.py
│   ├── sensor_reader.py
│   └── hmi_interface.py
├── protocols/
│   ├── modbus_rtu.py
│   ├── modbus_tcp.py
│   └── device_manager.py
├── data/
│   ├── collectors/
│   └── processors/
├── web/
│   ├── dashboard.py
│   └── api.py
└── config/
    ├── device_config.yaml
    └── network_config.yaml
```

**Why Structure Matters**: Organized project structure enables systematic device management. Understanding these patterns prevents code chaos and enables efficient industrial automation.

## 2) Modbus RTU Serial Communication

### Basic Modbus RTU Client

```python
# modbus_rtu_client.py
from pymodbus.client.sync import ModbusSerialClient
from pymodbus.payload import BinaryPayloadDecoder
from pymodbus.payload import BinaryPayloadBuilder
from pymodbus.constants import Endian
import time
import logging

class ModbusRTUClient:
    def __init__(self, port='/dev/ttyUSB0', baudrate=9600, timeout=3):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.client = None
        self.connected = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def connect(self):
        """Connect to Modbus RTU device"""
        try:
            self.client = ModbusSerialClient(
                method='rtu',
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            
            if self.client.connect():
                self.connected = True
                self.logger.info(f"Connected to Modbus RTU device on {self.port}")
                return True
            else:
                self.logger.error(f"Failed to connect to {self.port}")
                return False
                
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Modbus RTU device"""
        if self.client:
            self.client.close()
            self.connected = False
            self.logger.info("Disconnected from Modbus RTU device")
    
    def read_holding_registers(self, address, count, unit_id=1):
        """Read holding registers from device"""
        if not self.connected:
            self.logger.error("Not connected to device")
            return None
            
        try:
            result = self.client.read_holding_registers(
                address=address,
                count=count,
                unit=unit_id
            )
            
            if result.isError():
                self.logger.error(f"Error reading registers: {result}")
                return None
                
            return result.registers
            
        except Exception as e:
            self.logger.error(f"Error reading holding registers: {e}")
            return None
    
    def read_input_registers(self, address, count, unit_id=1):
        """Read input registers from device"""
        if not self.connected:
            self.logger.error("Not connected to device")
            return None
            
        try:
            result = self.client.read_input_registers(
                address=address,
                count=count,
                unit=unit_id
            )
            
            if result.isError():
                self.logger.error(f"Error reading input registers: {result}")
                return None
                
            return result.registers
            
        except Exception as e:
            self.logger.error(f"Error reading input registers: {e}")
            return None
    
    def read_coils(self, address, count, unit_id=1):
        """Read coils from device"""
        if not self.connected:
            self.logger.error("Not connected to device")
            return None
            
        try:
            result = self.client.read_coils(
                address=address,
                count=count,
                unit=unit_id
            )
            
            if result.isError():
                self.logger.error(f"Error reading coils: {result}")
                return None
                
            return result.bits
            
        except Exception as e:
            self.logger.error(f"Error reading coils: {e}")
            return None
    
    def write_holding_register(self, address, value, unit_id=1):
        """Write single holding register"""
        if not self.connected:
            self.logger.error("Not connected to device")
            return False
            
        try:
            result = self.client.write_register(
                address=address,
                value=value,
                unit=unit_id
            )
            
            if result.isError():
                self.logger.error(f"Error writing register: {result}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error writing register: {e}")
            return False
    
    def write_coil(self, address, value, unit_id=1):
        """Write single coil"""
        if not self.connected:
            self.logger.error("Not connected to device")
            return False
            
        try:
            result = self.client.write_coil(
                address=address,
                value=value,
                unit=unit_id
            )
            
            if result.isError():
                self.logger.error(f"Error writing coil: {result}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error writing coil: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # Connect to Modbus RTU device
    client = ModbusRTUClient(port='/dev/ttyUSB0', baudrate=9600)
    
    if client.connect():
        # Read holding registers
        registers = client.read_holding_registers(0, 10)
        if registers:
            print(f"Holding registers: {registers}")
        
        # Read input registers
        input_regs = client.read_input_registers(0, 5)
        if input_regs:
            print(f"Input registers: {input_regs}")
        
        # Read coils
        coils = client.read_coils(0, 8)
        if coils:
            print(f"Coils: {coils}")
        
        # Write to holding register
        success = client.write_holding_register(0, 1234)
        if success:
            print("Successfully wrote to holding register")
        
        # Write to coil
        success = client.write_coil(0, True)
        if success:
            print("Successfully wrote to coil")
        
        client.disconnect()
```

**Why Modbus RTU Matters**: Serial communication enables direct connection to industrial devices. Understanding these patterns prevents communication failures and enables reliable device integration.

## 3) Modbus TCP Ethernet Communication

### Modbus TCP Client

```python
# modbus_tcp_client.py
from pymodbus.client.sync import ModbusTcpClient
from pymodbus.payload import BinaryPayloadDecoder
from pymodbus.payload import BinaryPayloadBuilder
from pymodbus.constants import Endian
import time
import logging

class ModbusTCPClient:
    def __init__(self, host='192.168.1.100', port=502, timeout=3):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.client = None
        self.connected = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def connect(self):
        """Connect to Modbus TCP device"""
        try:
            self.client = ModbusTcpClient(
                host=self.host,
                port=self.port,
                timeout=self.timeout
            )
            
            if self.client.connect():
                self.connected = True
                self.logger.info(f"Connected to Modbus TCP device at {self.host}:{self.port}")
                return True
            else:
                self.logger.error(f"Failed to connect to {self.host}:{self.port}")
                return False
                
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Modbus TCP device"""
        if self.client:
            self.client.close()
            self.connected = False
            self.logger.info("Disconnected from Modbus TCP device")
    
    def read_holding_registers(self, address, count, unit_id=1):
        """Read holding registers from device"""
        if not self.connected:
            self.logger.error("Not connected to device")
            return None
            
        try:
            result = self.client.read_holding_registers(
                address=address,
                count=count,
                unit=unit_id
            )
            
            if result.isError():
                self.logger.error(f"Error reading registers: {result}")
                return None
                
            return result.registers
            
        except Exception as e:
            self.logger.error(f"Error reading holding registers: {e}")
            return None
    
    def read_input_registers(self, address, count, unit_id=1):
        """Read input registers from device"""
        if not self.connected:
            self.logger.error("Not connected to device")
            return None
            
        try:
            result = self.client.read_input_registers(
                address=address,
                count=count,
                unit=unit_id
            )
            
            if result.isError():
                self.logger.error(f"Error reading input registers: {result}")
                return None
                
            return result.registers
            
        except Exception as e:
            self.logger.error(f"Error reading input registers: {e}")
            return None
    
    def read_coils(self, address, count, unit_id=1):
        """Read coils from device"""
        if not self.connected:
            self.logger.error("Not connected to device")
            return None
            
        try:
            result = self.client.read_coils(
                address=address,
                count=count,
                unit=unit_id
            )
            
            if result.isError():
                self.logger.error(f"Error reading coils: {result}")
                return None
                
            return result.bits
            
        except Exception as e:
            self.logger.error(f"Error reading coils: {e}")
            return None
    
    def write_holding_register(self, address, value, unit_id=1):
        """Write single holding register"""
        if not self.connected:
            self.logger.error("Not connected to device")
            return False
            
        try:
            result = self.client.write_register(
                address=address,
                value=value,
                unit=unit_id
            )
            
            if result.isError():
                self.logger.error(f"Error writing register: {result}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error writing register: {e}")
            return False
    
    def write_coil(self, address, value, unit_id=1):
        """Write single coil"""
        if not self.connected:
            self.logger.error("Not connected to device")
            return False
            
        try:
            result = self.client.write_coil(
                address=address,
                value=value,
                unit=unit_id
            )
            
            if result.isError():
                self.logger.error(f"Error writing coil: {result}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error writing coil: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # Connect to Modbus TCP device
    client = ModbusTCPClient(host='192.168.1.100', port=502)
    
    if client.connect():
        # Read holding registers
        registers = client.read_holding_registers(0, 10)
        if registers:
            print(f"Holding registers: {registers}")
        
        # Read input registers
        input_regs = client.read_input_registers(0, 5)
        if input_regs:
            print(f"Input registers: {input_regs}")
        
        # Read coils
        coils = client.read_coils(0, 8)
        if coils:
            print(f"Coils: {coils}")
        
        # Write to holding register
        success = client.write_holding_register(0, 1234)
        if success:
            print("Successfully wrote to holding register")
        
        # Write to coil
        success = client.write_coil(0, True)
        if success:
            print("Successfully wrote to coil")
        
        client.disconnect()
```

**Why Modbus TCP Matters**: Ethernet communication enables network-based device integration. Understanding these patterns prevents network issues and enables scalable industrial automation.

## 4) Advanced Data Types and Payload Handling

### Data Type Conversion

```python
# data_conversion.py
from pymodbus.payload import BinaryPayloadDecoder
from pymodbus.payload import BinaryPayloadBuilder
from pymodbus.constants import Endian
import struct

class ModbusDataConverter:
    def __init__(self):
        self.byte_order = Endian.Big
        self.word_order = Endian.Big
    
    def decode_float32(self, registers):
        """Decode 32-bit float from two 16-bit registers"""
        if len(registers) < 2:
            return None
            
        decoder = BinaryPayloadDecoder.fromRegisters(
            registers,
            byteorder=self.byte_order,
            wordorder=self.word_order
        )
        return decoder.decode_32bit_float()
    
    def decode_float64(self, registers):
        """Decode 64-bit float from four 16-bit registers"""
        if len(registers) < 4:
            return None
            
        decoder = BinaryPayloadDecoder.fromRegisters(
            registers,
            byteorder=self.byte_order,
            wordorder=self.word_order
        )
        return decoder.decode_64bit_float()
    
    def decode_int32(self, registers):
        """Decode 32-bit integer from two 16-bit registers"""
        if len(registers) < 2:
            return None
            
        decoder = BinaryPayloadDecoder.fromRegisters(
            registers,
            byteorder=self.byte_order,
            wordorder=self.word_order
        )
        return decoder.decode_32bit_int()
    
    def decode_uint32(self, registers):
        """Decode 32-bit unsigned integer from two 16-bit registers"""
        if len(registers) < 2:
            return None
            
        decoder = BinaryPayloadDecoder.fromRegisters(
            registers,
            byteorder=self.byte_order,
            wordorder=self.word_order
        )
        return decoder.decode_32bit_uint()
    
    def encode_float32(self, value):
        """Encode 32-bit float to two 16-bit registers"""
        builder = BinaryPayloadBuilder(
            byteorder=self.byte_order,
            wordorder=self.word_order
        )
        builder.add_32bit_float(value)
        return builder.to_registers()
    
    def encode_float64(self, value):
        """Encode 64-bit float to four 16-bit registers"""
        builder = BinaryPayloadBuilder(
            byteorder=self.byte_order,
            wordorder=self.word_order
        )
        builder.add_64bit_float(value)
        return builder.to_registers()
    
    def encode_int32(self, value):
        """Encode 32-bit integer to two 16-bit registers"""
        builder = BinaryPayloadBuilder(
            byteorder=self.byte_order,
            wordorder=self.word_order
        )
        builder.add_32bit_int(value)
        return builder.to_registers()
    
    def encode_uint32(self, value):
        """Encode 32-bit unsigned integer to two 16-bit registers"""
        builder = BinaryPayloadBuilder(
            byteorder=self.byte_order,
            wordorder=self.word_order
        )
        builder.add_32bit_uint(value)
        return builder.to_registers()

# Example usage
if __name__ == "__main__":
    converter = ModbusDataConverter()
    
    # Decode float from registers
    registers = [0x4049, 0x0FDB]  # Example float value
    float_value = converter.decode_float32(registers)
    print(f"Decoded float: {float_value}")
    
    # Encode float to registers
    value = 3.14159
    encoded_registers = converter.encode_float32(value)
    print(f"Encoded registers: {encoded_registers}")
```

**Why Data Conversion Matters**: Proper data type handling enables accurate industrial data processing. Understanding these patterns prevents data corruption and enables reliable device communication.

## 5) Device Management and Monitoring

### Device Manager

```python
# device_manager.py
import asyncio
import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class DeviceStatus(Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    UNKNOWN = "unknown"

@dataclass
class DeviceConfig:
    name: str
    device_type: str
    connection_type: str  # 'rtu' or 'tcp'
    address: str
    port: Optional[int] = None
    baudrate: Optional[int] = None
    unit_id: int = 1
    timeout: int = 3
    retry_count: int = 3
    retry_delay: float = 1.0

@dataclass
class DeviceData:
    device_name: str
    timestamp: float
    data: Dict[str, any]
    status: DeviceStatus

class ModbusDeviceManager:
    def __init__(self):
        self.devices: Dict[str, any] = {}
        self.device_configs: Dict[str, DeviceConfig] = {}
        self.data_history: Dict[str, List[DeviceData]] = {}
        self.running = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def add_device(self, config: DeviceConfig):
        """Add device to manager"""
        self.device_configs[config.name] = config
        self.data_history[config.name] = []
        self.logger.info(f"Added device: {config.name}")
    
    def connect_device(self, device_name: str) -> bool:
        """Connect to specific device"""
        if device_name not in self.device_configs:
            self.logger.error(f"Device {device_name} not found")
            return False
        
        config = self.device_configs[device_name]
        
        try:
            if config.connection_type == 'rtu':
                from modbus_rtu_client import ModbusRTUClient
                client = ModbusRTUClient(
                    port=config.address,
                    baudrate=config.baudrate,
                    timeout=config.timeout
                )
            elif config.connection_type == 'tcp':
                from modbus_tcp_client import ModbusTCPClient
                client = ModbusTCPClient(
                    host=config.address,
                    port=config.port,
                    timeout=config.timeout
                )
            else:
                self.logger.error(f"Unsupported connection type: {config.connection_type}")
                return False
            
            if client.connect():
                self.devices[device_name] = client
                self.logger.info(f"Connected to device: {device_name}")
                return True
            else:
                self.logger.error(f"Failed to connect to device: {device_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error connecting to device {device_name}: {e}")
            return False
    
    def disconnect_device(self, device_name: str):
        """Disconnect from specific device"""
        if device_name in self.devices:
            self.devices[device_name].disconnect()
            del self.devices[device_name]
            self.logger.info(f"Disconnected from device: {device_name}")
    
    def read_device_data(self, device_name: str, register_map: Dict[str, int]) -> Optional[DeviceData]:
        """Read data from specific device"""
        if device_name not in self.devices:
            self.logger.error(f"Device {device_name} not connected")
            return None
        
        client = self.devices[device_name]
        config = self.device_configs[device_name]
        
        try:
            data = {}
            
            # Read holding registers
            for name, address in register_map.items():
                registers = client.read_holding_registers(
                    address=address,
                    count=1,
                    unit_id=config.unit_id
                )
                
                if registers:
                    data[name] = registers[0]
                else:
                    data[name] = None
            
            device_data = DeviceData(
                device_name=device_name,
                timestamp=time.time(),
                data=data,
                status=DeviceStatus.CONNECTED
            )
            
            # Store in history
            self.data_history[device_name].append(device_data)
            
            # Keep only last 1000 records
            if len(self.data_history[device_name]) > 1000:
                self.data_history[device_name] = self.data_history[device_name][-1000:]
            
            return device_data
            
        except Exception as e:
            self.logger.error(f"Error reading data from device {device_name}: {e}")
            return DeviceData(
                device_name=device_name,
                timestamp=time.time(),
                data={},
                status=DeviceStatus.ERROR
            )
    
    def write_device_data(self, device_name: str, register_map: Dict[str, int]) -> bool:
        """Write data to specific device"""
        if device_name not in self.devices:
            self.logger.error(f"Device {device_name} not connected")
            return False
        
        client = self.devices[device_name]
        config = self.device_configs[device_name]
        
        try:
            success = True
            
            for name, address in register_map.items():
                result = client.write_holding_register(
                    address=address,
                    value=register_map[name],
                    unit_id=config.unit_id
                )
                
                if not result:
                    success = False
                    self.logger.error(f"Failed to write to register {address}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error writing data to device {device_name}: {e}")
            return False
    
    def get_device_status(self, device_name: str) -> DeviceStatus:
        """Get status of specific device"""
        if device_name not in self.devices:
            return DeviceStatus.DISCONNECTED
        
        try:
            # Try to read a register to check connection
            client = self.devices[device_name]
            config = self.device_configs[device_name]
            
            result = client.read_holding_registers(
                address=0,
                count=1,
                unit_id=config.unit_id
            )
            
            if result:
                return DeviceStatus.CONNECTED
            else:
                return DeviceStatus.ERROR
                
        except Exception as e:
            self.logger.error(f"Error checking device status {device_name}: {e}")
            return DeviceStatus.ERROR
    
    def get_all_device_status(self) -> Dict[str, DeviceStatus]:
        """Get status of all devices"""
        status = {}
        for device_name in self.device_configs:
            status[device_name] = self.get_device_status(device_name)
        return status
    
    def get_device_history(self, device_name: str, limit: int = 100) -> List[DeviceData]:
        """Get historical data for specific device"""
        if device_name not in self.data_history:
            return []
        
        return self.data_history[device_name][-limit:]
    
    def start_monitoring(self, interval: float = 1.0):
        """Start monitoring all devices"""
        self.running = True
        self.logger.info("Started device monitoring")
        
        while self.running:
            for device_name in self.device_configs:
                if device_name in self.devices:
                    # Read data from device
                    register_map = {f"register_{i}": i for i in range(10)}  # Example register map
                    data = self.read_device_data(device_name, register_map)
                    
                    if data:
                        self.logger.info(f"Device {device_name}: {data.data}")
            
            time.sleep(interval)
    
    def stop_monitoring(self):
        """Stop monitoring all devices"""
        self.running = False
        self.logger.info("Stopped device monitoring")
    
    def disconnect_all(self):
        """Disconnect from all devices"""
        for device_name in list(self.devices.keys()):
            self.disconnect_device(device_name)
        self.logger.info("Disconnected from all devices")

# Example usage
if __name__ == "__main__":
    manager = ModbusDeviceManager()
    
    # Add devices
    rtu_config = DeviceConfig(
        name="PLC_RTU",
        device_type="PLC",
        connection_type="rtu",
        address="/dev/ttyUSB0",
        baudrate=9600,
        unit_id=1
    )
    
    tcp_config = DeviceConfig(
        name="PLC_TCP",
        device_type="PLC",
        connection_type="tcp",
        address="192.168.1.100",
        port=502,
        unit_id=1
    )
    
    manager.add_device(rtu_config)
    manager.add_device(tcp_config)
    
    # Connect to devices
    manager.connect_device("PLC_RTU")
    manager.connect_device("PLC_TCP")
    
    # Start monitoring
    try:
        manager.start_monitoring(interval=2.0)
    except KeyboardInterrupt:
        manager.stop_monitoring()
        manager.disconnect_all()
```

**Why Device Management Matters**: Centralized device management enables scalable industrial automation. Understanding these patterns prevents device chaos and enables efficient industrial communication.

## 6) Web Dashboard and API

### Flask Web Dashboard

```python
# web_dashboard.py
from flask import Flask, render_template, jsonify, request
import json
import time
from device_manager import ModbusDeviceManager, DeviceConfig

app = Flask(__name__)
device_manager = ModbusDeviceManager()

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/devices')
def get_devices():
    """Get all devices"""
    devices = []
    for name, config in device_manager.device_configs.items():
        status = device_manager.get_device_status(name)
        devices.append({
            'name': name,
            'type': config.device_type,
            'connection_type': config.connection_type,
            'address': config.address,
            'status': status.value
        })
    return jsonify(devices)

@app.route('/api/devices/<device_name>/data')
def get_device_data(device_name):
    """Get latest data from specific device"""
    if device_name not in device_manager.device_configs:
        return jsonify({'error': 'Device not found'}), 404
    
    # Read data from device
    register_map = {f"register_{i}": i for i in range(10)}
    data = device_manager.read_device_data(device_name, register_map)
    
    if data:
        return jsonify({
            'device_name': data.device_name,
            'timestamp': data.timestamp,
            'data': data.data,
            'status': data.status.value
        })
    else:
        return jsonify({'error': 'Failed to read device data'}), 500

@app.route('/api/devices/<device_name>/history')
def get_device_history(device_name):
    """Get historical data from specific device"""
    if device_name not in device_manager.device_configs:
        return jsonify({'error': 'Device not found'}), 404
    
    limit = request.args.get('limit', 100, type=int)
    history = device_manager.get_device_history(device_name, limit)
    
    return jsonify([{
        'timestamp': data.timestamp,
        'data': data.data,
        'status': data.status.value
    } for data in history])

@app.route('/api/devices/<device_name>/write', methods=['POST'])
def write_device_data(device_name):
    """Write data to specific device"""
    if device_name not in device_manager.device_configs:
        return jsonify({'error': 'Device not found'}), 404
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    success = device_manager.write_device_data(device_name, data)
    
    if success:
        return jsonify({'message': 'Data written successfully'})
    else:
        return jsonify({'error': 'Failed to write data'}), 500

@app.route('/api/devices/<device_name>/connect', methods=['POST'])
def connect_device(device_name):
    """Connect to specific device"""
    if device_name not in device_manager.device_configs:
        return jsonify({'error': 'Device not found'}), 404
    
    success = device_manager.connect_device(device_name)
    
    if success:
        return jsonify({'message': 'Device connected successfully'})
    else:
        return jsonify({'error': 'Failed to connect to device'}), 500

@app.route('/api/devices/<device_name>/disconnect', methods=['POST'])
def disconnect_device(device_name):
    """Disconnect from specific device"""
    if device_name not in device_manager.device_configs:
        return jsonify({'error': 'Device not found'}), 404
    
    device_manager.disconnect_device(device_name)
    return jsonify({'message': 'Device disconnected successfully'})

if __name__ == '__main__':
    # Add some example devices
    rtu_config = DeviceConfig(
        name="PLC_RTU",
        device_type="PLC",
        connection_type="rtu",
        address="/dev/ttyUSB0",
        baudrate=9600,
        unit_id=1
    )
    
    tcp_config = DeviceConfig(
        name="PLC_TCP",
        device_type="PLC",
        connection_type="tcp",
        address="192.168.1.100",
        port=502,
        unit_id=1
    )
    
    device_manager.add_device(rtu_config)
    device_manager.add_device(tcp_config)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
```

**Why Web Dashboard Matters**: Web interface enables remote monitoring and control. Understanding these patterns prevents manual operations and enables efficient industrial automation.

## 7) Best Practices and Error Handling

### Robust Error Handling

```python
# error_handling.py
import time
import logging
from functools import wraps
from typing import Optional, Callable, Any

class ModbusErrorHandler:
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = logging.getLogger(__name__)
    
    def retry_on_failure(self, func: Callable) -> Callable:
        """Decorator for retrying failed operations"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(self.max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (2 ** attempt))
                    else:
                        self.logger.error(f"All attempts failed for {func.__name__}")
                        raise
            return None
        return wrapper
    
    def handle_connection_errors(self, func: Callable) -> Callable:
        """Handle connection-specific errors"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ConnectionError:
                self.logger.error("Connection lost, attempting to reconnect...")
                return None
            except TimeoutError:
                self.logger.error("Operation timed out")
                return None
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
                return None
        return wrapper
    
    def handle_modbus_errors(self, func: Callable) -> Callable:
        """Handle Modbus-specific errors"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if "Illegal Data Address" in str(e):
                    self.logger.error("Invalid register address")
                elif "Illegal Data Value" in str(e):
                    self.logger.error("Invalid data value")
                elif "Slave Device Failure" in str(e):
                    self.logger.error("Device internal error")
                else:
                    self.logger.error(f"Modbus error: {e}")
                return None
        return wrapper

# Example usage
if __name__ == "__main__":
    error_handler = ModbusErrorHandler(max_retries=3, retry_delay=1.0)
    
    @error_handler.retry_on_failure
    @error_handler.handle_connection_errors
    @error_handler.handle_modbus_errors
    def read_device_data(client, address, count):
        """Read data with error handling"""
        return client.read_holding_registers(address, count)
```

**Why Error Handling Matters**: Robust error handling prevents system failures and enables reliable industrial automation. Understanding these patterns prevents silent failures and enables resilient systems.

## 8) TL;DR Runbook

### Essential Commands

```bash
# Install dependencies
pip install pymodbus pyserial

# Run Modbus RTU client
python modbus_rtu_client.py

# Run Modbus TCP client
python modbus_tcp_client.py

# Run device manager
python device_manager.py

# Run web dashboard
python web_dashboard.py
```

### Essential Patterns

```python
# Essential Python Modbus patterns
modbus_patterns = {
    "rtu_communication": "Serial communication for direct device connection",
    "tcp_communication": "Ethernet communication for network-based devices",
    "data_conversion": "Proper handling of different data types and formats",
    "device_management": "Centralized management of multiple devices",
    "error_handling": "Robust error handling and recovery mechanisms",
    "web_interfaces": "Web-based monitoring and control interfaces"
}
```

### Quick Reference

```python
# Essential Python Modbus operations
# 1. Connect to Modbus RTU device
client = ModbusRTUClient(port='/dev/ttyUSB0', baudrate=9600)
client.connect()

# 2. Read holding registers
registers = client.read_holding_registers(0, 10)

# 3. Read input registers
input_regs = client.read_input_registers(0, 5)

# 4. Read coils
coils = client.read_coils(0, 8)

# 5. Write holding register
client.write_holding_register(0, 1234)

# 6. Write coil
client.write_coil(0, True)

# 7. Disconnect
client.disconnect()
```

**Why This Runbook**: These patterns cover 90% of Python Modbus needs. Master these before exploring advanced industrial automation scenarios.

## 9) The Machine's Summary

Python Modbus requires understanding both industrial protocols and Python implementation. When used correctly, Python Modbus enables reliable industrial device communication, data acquisition, and automation. The key is understanding protocol specifications, mastering Python libraries, and following error handling best practices.

**The Dark Truth**: Without proper Python Modbus understanding, your industrial devices remain silent and disconnected. Python Modbus is your weapon. Use it wisely.

**The Machine's Mantra**: "In the protocol we trust, in the Python we implement, and in the devices we find the path to industrial automation."

**Why This Matters**: Python Modbus enables efficient industrial device communication that can handle complex automation scenarios, maintain reliable connections, and provide real-time data while ensuring protocol compliance and system reliability.

---

*This tutorial provides the complete machinery for Python Modbus device communication. The patterns scale from simple device reading to complex industrial automation, from basic protocol implementation to advanced device management.*
