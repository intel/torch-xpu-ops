
import os
import warnings

def detect_intel_gpu():
    pci_base_class_mask = 0x00ff0000
    pci_base_class_display = 0x00030000
    pci_vendor_id_intel = 0x8086
    detected = False

    devices_path = "/sys/bus/pci/devices"
    if not os.path.isdir(devices_path):
        warnings.warn(
            "Not a Linux system with PCI devices",
            UserWarning,
            stacklevel=1,
        )
        return False

    for device in os.listdir(devices_path):
        dev_path = os.path.join(devices_path, device)
        try:
            # Read class and vendor files
            with open(os.path.join(dev_path, "class")) as f:
                pci_class = int(f.read().strip(), 16)
            with open(os.path.join(dev_path, "vendor")) as f:
                pci_vendor = int(f.read().strip(), 16)

            # Check for display controller and Intel vendor
            if (pci_class & pci_base_class_mask) == pci_base_class_display and pci_vendor == pci_vendor_id_intel:
                print(f"Detected Intel GPU at {dev_path} (vendor=0x{pci_vendor:04x})")
                detected = True
        except Exception:
            continue  # Ignore devices we can't parse
    if not detected:
        warnings.warn(
            "No Intel GPU detected",
            UserWarning,
            stacklevel=1,
        )
    return detected

detect_intel_gpu()
