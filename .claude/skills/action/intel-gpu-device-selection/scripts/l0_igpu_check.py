"""Check if Level Zero GPU devices are iGPU or dGPU using ZE_DEVICE_PROPERTY_FLAG_INTEGRATED."""
import ctypes
import sys
from ctypes import c_uint32, c_uint64, c_char, c_void_p, byref, Structure, POINTER

ze = ctypes.CDLL("libze_loader.so.1")

# Proper function signatures
ze.zeInit.argtypes = [c_uint32]
ze.zeInit.restype = c_uint32
ze.zeDriverGet.argtypes = [POINTER(c_uint32), c_void_p]
ze.zeDriverGet.restype = c_uint32
ze.zeDeviceGet.argtypes = [c_void_p, POINTER(c_uint32), c_void_p]
ze.zeDeviceGet.restype = c_uint32

ZE_DEVICE_TYPE_GPU = 1
ZE_DEVICE_PROPERTY_FLAG_INTEGRATED = 0x1
ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES = 0x2


class ze_device_properties_t(Structure):
    _fields_ = [
        ("stype", c_uint32),
        # ctypes aligns pNext to 8 bytes automatically on 64-bit
        ("pNext", c_void_p),
        ("type", c_uint32),
        ("vendorId", c_uint32),
        ("deviceId", c_uint32),
        ("flags", c_uint32),
        ("subdeviceId", c_uint32),
        ("coreClockRate", c_uint32),
        ("maxMemAllocSize", c_uint64),
        ("maxHardwareContexts", c_uint32),
        ("maxCommandQueuePriority", c_uint32),
        ("numThreadsPerEU", c_uint32),
        ("physicalEUSimdWidth", c_uint32),
        ("numEUsPerSubslice", c_uint32),
        ("numSubslicesPerSlice", c_uint32),
        ("numSlices", c_uint32),
        ("timerResolution", c_uint64),
        ("timestampValidBits", c_uint32),
        ("kernelTimestampValidBits", c_uint32),
        ("uuid", c_char * 16),
        ("name", c_char * 256),
    ]


ze.zeDeviceGetProperties.argtypes = [c_void_p, POINTER(ze_device_properties_t)]
ze.zeDeviceGetProperties.restype = c_uint32

rc = ze.zeInit(c_uint32(0))
if rc != 0:
    print(f"zeInit failed: 0x{rc:x}")
    sys.exit(1)

driver_count = c_uint32(0)
ze.zeDriverGet(byref(driver_count), None)
drivers = (c_void_p * driver_count.value)()
ze.zeDriverGet(byref(driver_count), ctypes.cast(drivers, c_void_p))

for i in range(driver_count.value):
    drv = drivers[i]
    dev_count = c_uint32(0)
    ze.zeDeviceGet(c_void_p(drv), byref(dev_count), None)
    devices = (c_void_p * dev_count.value)()
    ze.zeDeviceGet(c_void_p(drv), byref(dev_count), ctypes.cast(devices, c_void_p))

    for j in range(dev_count.value):
        props = ze_device_properties_t()
        props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES
        props.pNext = None
        ze.zeDeviceGetProperties(c_void_p(devices[j]), byref(props))

        if props.type == ZE_DEVICE_TYPE_GPU:
            integrated = bool(props.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED)
            label = "iGPU" if integrated else "dGPU"
            name = props.name.decode().strip()
            print(f"{name:40s} {label}  (devId=0x{props.deviceId:04x})")
