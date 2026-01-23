# conftest.py - Enhanced worker management with XPU memory clearing and test rerun
import pytest
import os
import sys
import time
import gc
import atexit
from typing import Optional, Dict, Any, Tuple
import threading
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== Configuration ====================
# Use environment variables for configuration to avoid recursion
_ENABLE_XPU_MEMORY_CLEARING = os.environ.get('ENABLE_XPU_MEMORY_CLEARING', '1') == '1'
_ENABLE_TEST_RERUN = os.environ.get('ENABLE_TEST_RERUN', '1') == '1'
_WORKER_RESTART_EXIT_CODE = 101

# Test rerun configuration
_RERUN_PATTERNS = [
    # (pattern_lower, pattern_name, max_retries)
    ('error_device_lost', 'UR_RESULT_ERROR_DEVICE_LOST', 0),
    ('crashed while running', 'Worker crash', 2),
    ('xpu error', 'XPU error', 0),
    ('gpu error', 'GPU error', 0),
    ('outofmemoryerror', 'Out of memory', 1),
    ('out_of_memory', 'Out of memory', 1),
    ('out of memory', 'Out of memory', 1),
    ('oom', 'Out of memory', 1),
    ('illegal memory', 'Illegal memory access', 0),
    ('device-side assert', 'Device-side assert', 0),
    ('segmentation fault', 'Segmentation fault', 0),  # 0 means restart worker
    ('bus error', 'Bus error', 0),  # 0 means restart worker
    ('kernel died', 'Kernel died', 0),  # 0 means restart worker
]

# ==================== Global State ====================
_worker_states: Dict[str, Dict[str, Any]] = {}
_current_worker_id: Optional[str] = None
_initialized = False

# Test rerun tracking
_test_rerun_tracker: Dict[str, Dict[str, Any]] = {}  # test_id -> {attempts: int, last_error: str, patterns_matched: List[str]}
_rerun_lock = threading.Lock()

# ==================== Safe Logger ====================
class SafeLogger:
    """Logger that handles closed streams gracefully"""

    @staticmethod
    def info(msg: str, *args, **kwargs):
        """Safely log info message"""
        try:
            logger.info(msg, *args, **kwargs)
        except (ValueError, OSError) as e:
            # Ignore logging errors during cleanup
            if "closed" not in str(e).lower():
                print(f"[LOGGER ERROR] {e}: {msg[:100]}...", file=sys.stderr)

    @staticmethod
    def warning(msg: str, *args, **kwargs):
        """Safely log warning message"""
        try:
            logger.warning(msg, *args, **kwargs)
        except (ValueError, OSError) as e:
            if "closed" not in str(e).lower():
                print(f"[LOGGER ERROR] {e}: {msg[:100]}...", file=sys.stderr)

    @staticmethod
    def error(msg: str, *args, **kwargs):
        """Safely log error message"""
        try:
            logger.error(msg, *args, **kwargs)
        except (ValueError, OSError) as e:
            if "closed" not in str(e).lower():
                print(f"[LOGGER ERROR] {e}: {msg[:100]}...", file=sys.stderr)

# ==================== XPU Memory Manager ====================
class XPUMemoryManager:
    """Manage XPU memory operations"""

    _lock = threading.Lock()

    @staticmethod
    def has_xpu() -> bool:
        """Check if XPU is available"""
        try:
            return hasattr(torch, 'xpu') and callable(getattr(torch, 'xpu', None)) and torch.xpu.is_available()
        except (ImportError, AttributeError, RuntimeError):
            return False

    @staticmethod
    def silent_clear_xpu_memory() -> None:
        """Clear XPU memory without any logging"""
        if not _ENABLE_XPU_MEMORY_CLEARING:
            return

        with XPUMemoryManager._lock:
            try:
                # Force Python garbage collection first
                gc.collect()

                # Clear XPU memory if available
                if XPUMemoryManager.has_xpu():
                    try:
                        torch.xpu.synchronize()
                        torch.xpu.empty_cache()
                    except Exception:
                        pass  # Silent cleanup

            except ImportError:
                pass  # torch not installed
            except Exception:
                pass  # Silent cleanup


# ==================== Test Rerun Manager ====================
class TestRerunManager:
    """Manage test rerun logic based on error patterns"""

    @staticmethod
    def get_test_id(item) -> str:
        """Get unique test identifier"""
        return f"{item.nodeid}"

    @staticmethod
    def should_rerun_test(item, error_msg: str) -> Tuple[bool, Optional[str], int]:
        """
        Check if test should be rerun based on error patterns.
        Returns: (should_rerun, pattern_name, max_retries)
        """
        if not _ENABLE_TEST_RERUN:
            return False, None, 0

        test_id = TestRerunManager.get_test_id(item)

        with _rerun_lock:
            # Get or create test tracking entry
            if test_id not in _test_rerun_tracker:
                _test_rerun_tracker[test_id] = {
                    'attempts': 0,
                    'last_error': '',
                    'patterns_matched': [],
                    'max_attempts': 0
                }

            tracker = _test_rerun_tracker[test_id]
            tracker['attempts'] += 1
            tracker['last_error'] = error_msg

            # Check error against patterns
            error_lower = error_msg.lower()
            matched_patterns = []

            for pattern_lower, pattern_name, max_retries in _RERUN_PATTERNS:
                if pattern_lower in error_lower:
                    matched_patterns.append(pattern_name)

                    # If max_retries is 0, this is a fatal error that requires worker restart
                    if max_retries == 0:
                        return False, pattern_name, 0

                    # Check if we have retries left
                    if tracker['attempts'] <= max_retries + 1:  # +1 for initial attempt
                        tracker['max_attempts'] = max_retries
                        tracker['patterns_matched'].append(pattern_name)
                        return True, pattern_name, max_retries

            # No matching patterns or retries exhausted
            return False, None, 0

    @staticmethod
    def get_rerun_summary() -> Dict[str, Any]:
        """Get summary of all rerun attempts"""
        with _rerun_lock:
            total_reruns = 0
            successful_reruns = 0
            failed_after_rerun = 0

            for tracker in _test_rerun_tracker.values():
                if tracker['attempts'] > 1:
                    total_reruns += tracker['attempts'] - 1
                    if tracker.get('final_success', False):
                        successful_reruns += 1
                    else:
                        failed_after_rerun += 1

            return {
                'total_tests_rerun': len([t for t in _test_rerun_tracker.values() if t['attempts'] > 1]),
                'total_rerun_attempts': total_reruns,
                'successful_reruns': successful_reruns,
                'failed_after_rerun': failed_after_rerun,
                'tests_rerun': _test_rerun_tracker
            }

    @staticmethod
    def mark_test_success(item):
        """Mark test as successful after rerun"""
        test_id = TestRerunManager.get_test_id(item)
        with _rerun_lock:
            if test_id in _test_rerun_tracker:
                _test_rerun_tracker[test_id]['final_success'] = True

    @staticmethod
    def clear_test_tracking(item):
        """Clear tracking for a specific test"""
        test_id = TestRerunManager.get_test_id(item)
        with _rerun_lock:
            if test_id in _test_rerun_tracker:
                del _test_rerun_tracker[test_id]


# ==================== Utility Functions ====================
def get_worker_id(config) -> Optional[str]:
    """Get worker ID from config"""
    if hasattr(config, 'workerinput'):
        return config.workerinput.get('workerid')
    return None

def is_worker_process(config) -> bool:
    """Check if current process is a worker"""
    return hasattr(config, 'workerinput')

def restart_worker(worker_id: str, reason: str) -> None:
    """Restart worker with given reason"""

    SafeLogger.info(f"\n{'='*60}")
    SafeLogger.info(f"🔄 RESTARTING WORKER {worker_id}")
    SafeLogger.info(f"   Reason: {reason}")
    SafeLogger.info(f"   Time: {time.strftime('%H:%M:%S')}")
    SafeLogger.info(f"{'='*60}")

    # Flush output
    sys.stdout.flush()
    sys.stderr.flush()

    # Force memory cleanup before exit
    XPUMemoryManager.silent_clear_xpu_memory()

    # Exit with special code
    os._exit(_WORKER_RESTART_EXIT_CODE)

# ==================== Cleanup Handler ====================
class CleanupManager:
    """Manage cleanup operations with proper resource handling"""

    _registered = False

    @staticmethod
    def register_cleanup():
        """Register cleanup handlers"""
        if CleanupManager._registered:
            return

        # Register cleanup at exit
        atexit.register(CleanupManager.cleanup)
        CleanupManager._registered = True

    @staticmethod
    def cleanup():
        """Cleanup function that handles closed streams gracefully"""
        if not _current_worker_id:
            return

        try:
            # Silent memory cleanup
            XPUMemoryManager.silent_clear_xpu_memory()

        except Exception:
            pass  # Final cleanup should never raise

# ==================== Pytest Hooks ====================
def _pytest_configure_impl(config):
    """Actual configure implementation"""
    global _current_worker_id, _initialized

    if _initialized:
        return

    if is_worker_process(config):
        _current_worker_id = get_worker_id(config)

        if _current_worker_id:
            _worker_states[_current_worker_id] = {
                'start_time': time.time(),
                'memory_clears': 0,
                'last_memory_check': time.time(),
                'tests_rerun': 0,
                'tests_rerun_success': 0
            }

            SafeLogger.info(f"\n{'='*60}")
            SafeLogger.info(f"🚀 WORKER {_current_worker_id} INITIALIZED")
            SafeLogger.info(f"   XPU memory clearing: {_ENABLE_XPU_MEMORY_CLEARING}")
            SafeLogger.info(f"   Test rerun enabled: {_ENABLE_TEST_RERUN}")
            if XPUMemoryManager.has_xpu():
                SafeLogger.info("   XPU available: Yes")
            SafeLogger.info(f"{'='*60}\n")

            # Initial memory clearing
            XPUMemoryManager.silent_clear_xpu_memory()

            # Register cleanup handler
            CleanupManager.register_cleanup()

    _initialized = True

@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    """Configure worker on startup"""
    _pytest_configure_impl(config)

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    """Handle test execution with rerun capability"""

    # Clear previous tracking if this is a fresh run
    if not hasattr(item, '_already_rerun'):
        TestRerunManager.clear_test_tracking(item)

    # Execute test with possible reruns
    attempts = 0

    while attempts < 3:  # Safety limit
        attempts += 1

        # Run the test
        try:
            # Call the original runtest protocol
            yield
            break  # Test passed, exit loop

        except Exception as e:
            # Test failed, check if we should rerun
            error_msg = str(e)
            should_rerun, pattern_name, max_retries = TestRerunManager.should_rerun_test(item, error_msg)

            if should_rerun and attempts <= max_retries + 1:
                SafeLogger.info(f"\n{'='*60}")
                SafeLogger.info(f"🔄 RERUNNING TEST: {item.name}")
                SafeLogger.info(f"   Attempt: {attempts}/{max_retries + 1}")
                SafeLogger.info(f"   Error pattern: {pattern_name}")
                SafeLogger.info(f"   Error: {error_msg[:200]}...")
                SafeLogger.info(f"{'='*60}")

                # Clear memory before rerun
                XPUMemoryManager.silent_clear_xpu_memory()

                # Add delay before rerun
                time.sleep(1)

                # Mark that we're rerunning
                item._already_rerun = True

                # Update worker stats
                if _current_worker_id in _worker_states:
                    _worker_states[_current_worker_id]['tests_rerun'] += 1

                continue  # Try again
            else:
                # No rerun or max attempts reached, re-raise the exception
                raise

    # Mark success if we passed after rerun
    if attempts > 1:
        TestRerunManager.mark_test_success(item)
        if _current_worker_id in _worker_states:
            _worker_states[_current_worker_id]['tests_rerun_success'] += 1

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    """Clear XPU memory before each test"""
    if _current_worker_id and _ENABLE_XPU_MEMORY_CLEARING:
        XPUMemoryManager.silent_clear_xpu_memory()

    yield  # Run the actual test setup

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """Wrap test execution"""
    yield  # Run the actual test

    if _current_worker_id and _ENABLE_XPU_MEMORY_CLEARING:
        XPUMemoryManager.silent_clear_xpu_memory()

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    """Clear XPU memory after test teardown"""
    yield  # Run the actual teardown

    if _current_worker_id and _ENABLE_XPU_MEMORY_CLEARING:
        XPUMemoryManager.silent_clear_xpu_memory()

        if _current_worker_id in _worker_states:
            _worker_states[_current_worker_id]['memory_clears'] += 1

@pytest.hookimpl
def pytest_runtest_logreport(report):
    """Monitor test results and restart on failures"""

    if not _current_worker_id:
        return
    if not report.failed:
        return
    if _current_worker_id not in _worker_states:
        return

    # Check for fatal device errors
    fatal_error = False
    fatal_pattern = None

    if report.longrepr:
        error_msg = str(report.longrepr).lower()

        # Check patterns for fatal errors (max_retries == 0)
        for pattern_lower, pattern_name, max_retries in _RERUN_PATTERNS:
            if max_retries == 0 and pattern_lower in error_msg:
                fatal_error = True
                fatal_pattern = pattern_name
                break

    # Check if we need to restart
    if fatal_error:
        reason = fatal_pattern

        # Clear memory before restart
        XPUMemoryManager.silent_clear_xpu_memory()

        # Restart worker
        restart_worker(_current_worker_id, reason)

@pytest.hookimpl
def pytest_sessionfinish(session, exitstatus):
    """Print worker statistics at session end"""
    if _current_worker_id and _current_worker_id in _worker_states:
        state = _worker_states[_current_worker_id]
        duration = time.time() - state['start_time']

        # Get rerun summary
        rerun_summary = TestRerunManager.get_rerun_summary()

        SafeLogger.info(f"\n{'='*60}")
        SafeLogger.info(f"📊 WORKER {_current_worker_id} SUMMARY")
        SafeLogger.info(f"{'='*60}")
        SafeLogger.info(f"   Runtime:           {duration:.1f}s")
        SafeLogger.info(f"   Memory clears:     {state['memory_clears']}")

        if _ENABLE_TEST_RERUN:
            SafeLogger.info("\n   Test Rerun Stats:")
            SafeLogger.info(f"   Tests rerun:       {rerun_summary['total_tests_rerun']}")
            SafeLogger.info(f"   Rerun attempts:    {rerun_summary['total_rerun_attempts']}")
            SafeLogger.info(f"   Successful reruns: {rerun_summary['successful_reruns']}")
            SafeLogger.info(f"   Failed after retry:{rerun_summary['failed_after_rerun']}")

        SafeLogger.info(f"{'='*60}")

@pytest.hookimpl
def pytest_keyboard_interrupt(excinfo):
    """Handle keyboard interrupt"""
    if _current_worker_id:
        try:
            SafeLogger.info(f"\n⚠️  Keyboard interrupt in worker {_current_worker_id}")
        except (ValueError, OSError):
            print(f"\n⚠️  Keyboard interrupt in worker {_current_worker_id}", file=sys.stderr)
        CleanupManager.cleanup()

# ==================== Pytest Fixtures ====================
@pytest.fixture(scope="function", autouse=True)
def auto_clear_memory():
    """Automatically clear XPU memory after each test"""
    yield
    if _ENABLE_XPU_MEMORY_CLEARING and _current_worker_id:
        XPUMemoryManager.silent_clear_xpu_memory()

@pytest.fixture(scope="function")
def silent_clear_xpu_memory():
    """Fixture to manually clear XPU memory"""
    def clear():
        XPUMemoryManager.silent_clear_xpu_memory()
    return clear

@pytest.fixture(scope="function")
def rerun_on_error():
    """Fixture to manually trigger test rerun"""
    def should_rerun(error_msg: str, max_attempts: int = 2) -> bool:
        """Check if test should be rerun based on error message"""
        # You can add custom logic here
        return "temporary" in error_msg.lower() or "timeout" in error_msg.lower()

    return should_rerun

# ==================== Worker Exit Handler ====================
def handle_worker_exit():
    """Handle worker exit gracefully"""
    try:
        CleanupManager.cleanup()
    except Exception:
        pass  # Ensure we always exit cleanly

# Register exit handler
atexit.unregister(CleanupManager.cleanup)  # Remove if already registered
atexit.register(handle_worker_exit)
