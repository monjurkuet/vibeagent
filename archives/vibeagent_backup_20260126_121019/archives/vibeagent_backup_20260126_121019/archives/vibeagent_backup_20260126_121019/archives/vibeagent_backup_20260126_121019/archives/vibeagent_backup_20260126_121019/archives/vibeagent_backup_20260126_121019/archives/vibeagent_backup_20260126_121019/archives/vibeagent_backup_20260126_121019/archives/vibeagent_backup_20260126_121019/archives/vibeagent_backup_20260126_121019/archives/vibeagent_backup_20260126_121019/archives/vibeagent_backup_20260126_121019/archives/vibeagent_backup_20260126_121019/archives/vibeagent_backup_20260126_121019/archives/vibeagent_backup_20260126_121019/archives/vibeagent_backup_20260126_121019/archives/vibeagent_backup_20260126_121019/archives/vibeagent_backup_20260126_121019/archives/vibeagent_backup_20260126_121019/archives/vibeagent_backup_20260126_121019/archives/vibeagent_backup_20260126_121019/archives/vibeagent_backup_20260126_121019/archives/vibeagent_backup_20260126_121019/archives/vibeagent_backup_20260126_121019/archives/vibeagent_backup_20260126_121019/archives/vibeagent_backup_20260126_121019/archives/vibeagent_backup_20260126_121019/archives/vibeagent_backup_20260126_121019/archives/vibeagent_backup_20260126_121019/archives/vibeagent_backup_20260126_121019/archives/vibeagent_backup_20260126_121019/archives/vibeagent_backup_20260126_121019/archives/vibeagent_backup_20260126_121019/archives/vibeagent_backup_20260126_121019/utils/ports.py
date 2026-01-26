#!/usr/bin/env python3
"""
Port management utility for VibeAgent.
"""

import socket
import subprocess
import json
from pathlib import Path


def is_port_in_use(port: int) -> bool:
    """Check if a port is in use."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex(('localhost', port))
            return result == 0
    except:
        return False


def find_free_port(start: int = 8000, end: int = 9000) -> int:
    """Find a free port in the given range."""
    for port in range(start, end):
        if not is_port_in_use(port):
            return port
    return None


def get_port_status(port: int) -> dict:
    """Get detailed status of a port."""
    in_use = is_port_in_use(port)
    
    if in_use:
        try:
            result = subprocess.run(
                ['lsof', '-i', f':{port}', '-t'],
                capture_output=True,
                text=True
            )
            pid = result.stdout.strip() if result.stdout else None
            
            if pid:
                try:
                    process_info = subprocess.run(
                        ['ps', '-p', pid, '-o', 'comm='],
                        capture_output=True,
                        text=True
                    )
                    process_name = process_info.stdout.strip()
                except:
                    process_name = "unknown"
            else:
                process_name = "unknown"
        except:
            pid = None
            process_name = "unknown"
    else:
        pid = None
        process_name = None
    
    return {
        "port": port,
        "in_use": in_use,
        "pid": pid,
        "process": process_name
    }


def check_port_block(start: int, count: int = 10) -> list:
    """Check a block of ports."""
    results = []
    for port in range(start, start + count):
        results.append(get_port_status(port))
    return results


def main():
    """Main function."""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "check":
            # Check specific port
            port = int(sys.argv[2]) if len(sys.argv) > 2 else 8001
            status = get_port_status(port)
            print(f"Port {port}: {'IN USE' if status['in_use'] else 'FREE'}")
            if status['in_use']:
                print(f"  PID: {status['pid']}")
                print(f"  Process: {status['process']}")
        
        elif command == "find":
            # Find free port
            start = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
            port = find_free_port(start)
            if port:
                print(f"Found free port: {port}")
            else:
                print(f"No free port found in range {start}-{start+100}")
        
        elif command == "block":
            # Check port block
            start = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
            count = int(sys.argv[3]) if len(sys.argv) > 3 else 10
            results = check_port_block(start, count)
            
            print(f"Port Block {start}-{start+count-1}:")
            for status in results:
                status_str = "IN USE" if status['in_use'] else "FREE"
                print(f"  {status['port']}: {status_str}")
        
        else:
            print("Unknown command")
            print("Available commands:")
            print("  check <port>  - Check if port is in use")
            print("  find <start>  - Find free port starting from start")
            print("  block <start> <count>  - Check a block of ports")
    else:
        # Show default port status
        print("VibeAgent Port Status")
        print("=" * 40)
        
        # Load port config
        ports_path = Path("config/ports.json")
        if ports_path.exists():
            with open(ports_path, 'r') as f:
                ports_config = json.load(f)
            
            print("\nConfigured Ports:")
            for service, port in ports_config.get('ports', {}).items():
                status = get_port_status(port)
                status_str = "✓ FREE" if not status['in_use'] else "✗ IN USE"
                print(f"  {service:15} : {port:5} ({status_str})")
        else:
            print("No port configuration found")
            print("Using default ports:")
            print("  API: 8001")
            print("  Frontend: 3000")
            print("  WebSocket: 8001")
            print("  PocketBase: 8090")


if __name__ == "__main__":
    main()