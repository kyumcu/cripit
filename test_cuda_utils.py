""" 
Test suite for CUDA utilities module.

These tests are best-effort and designed to run on systems with or without CUDA.
They validate that detection logic is robust and does not crash.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import contextlib
import tempfile
import types
from unittest import mock

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@contextlib.contextmanager
def _temp_environ(updates):
    """Temporarily set environment variables."""
    old = {}
    try:
        for k, v in updates.items():
            old[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextlib.contextmanager
def _patched_modules(mods):
    """Temporarily patch sys.modules entries."""
    old = {}
    try:
        for name, module in mods.items():
            old[name] = sys.modules.get(name)
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        yield
    finally:
        for name, prev in old.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev


def test_cuda_detection_runs():
    """Test that CUDA detection runs without crashing."""
    print("\n" + "=" * 50)
    print("TEST 1: CUDA Detection Runs")
    print("=" * 50)

    from utils.cuda_utils import CUDAManager

    cm = CUDAManager()
    status = cm.detect_cuda(force_check=True)

    print(f"✓ available: {status.available}")
    print(f"✓ cuda_home: {status.cuda_home}")
    print(f"✓ nvcc_path: {status.nvcc_path}")
    print(f"✓ cuda_version: {status.cuda_version}")
    print(f"✓ devices: {len(status.devices)}")
    print(f"✓ pywhispercpp_cuda (best-effort): {status.pywhispercpp_cuda}")

    assert isinstance(status.available, bool)
    assert hasattr(status, "devices")
    return True


def test_gpu_memory_check_is_safe():
    """Test that GPU memory check does not crash even without torch/NVML."""
    print("\n" + "=" * 50)
    print("TEST 2: GPU Memory Check Safe")
    print("=" * 50)

    from utils.cuda_utils import CUDAManager

    cm = CUDAManager()
    mem = cm.get_gpu_memory_info(device_id=0)

    if mem is None:
        print("✓ Memory info unavailable (expected on systems without torch/NVML)")
    else:
        free_mb, total_mb = mem
        print(f"✓ free_mb: {free_mb}")
        print(f"✓ total_mb: {total_mb}")
        assert free_mb >= 0
        assert total_mb > 0

    return True


def test_model_fit_check_returns_message():
    """Test model fit check always returns (bool, str)."""
    print("\n" + "=" * 50)
    print("TEST 3: Model Fit Check")
    print("=" * 50)

    from utils.cuda_utils import CUDAManager

    cm = CUDAManager()
    fits, msg = cm.check_model_fits_gpu("large-v3-turbo", device_id=0)

    print(f"✓ fits: {fits}")
    print(f"✓ msg: {msg}")
    assert isinstance(fits, bool)
    assert isinstance(msg, str)
    assert len(msg) > 0
    return True


def test_detect_cuda_uses_nvidia_smi_without_torch():
    """Detect CUDA as available based on nvidia-smi, even without torch/NVML."""
    print("\n" + "=" * 50)
    print("TEST 4: Detect CUDA via nvidia-smi")
    print("=" * 50)

    from utils.cuda_utils import CUDAManager

    with tempfile.TemporaryDirectory() as td:
        cuda_home = os.path.join(td, "cuda")
        os.makedirs(os.path.join(cuda_home, "lib64"), exist_ok=True)

        def fake_run(cmd, capture_output=True, text=True, timeout=5, **kwargs):
            # Minimal emulation for detect_cuda() and _check_nvidia_smi()
            if cmd == ['which', 'nvcc']:
                return types.SimpleNamespace(returncode=0, stdout='/usr/bin/nvcc\n', stderr='')
            if cmd[:2] == ['/usr/bin/nvcc', '--version']:
                out = "Cuda compilation tools, release 12.0, V12.0.140\n"
                return types.SimpleNamespace(returncode=0, stdout=out, stderr='')
            if cmd[:2] == ['nvidia-smi', '--query-gpu=name']:
                return types.SimpleNamespace(returncode=0, stdout='FakeGPU 0\n', stderr='')
            return types.SimpleNamespace(returncode=1, stdout='', stderr='')

        with _temp_environ({'CUDA_HOME': cuda_home, 'CUDA_PATH': None}):
            with mock.patch('utils.cuda_utils.subprocess.run', side_effect=fake_run):
                cm = CUDAManager()
                # Avoid depending on the real pywhispercpp install in this unit test
                with mock.patch.object(cm, '_check_pywhispercpp_cuda', return_value=False):
                    status = cm.detect_cuda(force_check=True)

        print(f"✓ available: {status.available}")
        print(f"✓ cuda_home: {status.cuda_home}")
        print(f"✓ nvcc_path: {status.nvcc_path}")
        print(f"✓ cuda_version: {status.cuda_version}")
        print(f"✓ devices: {len(status.devices)}")
        assert status.available is True
        assert status.cuda_home == cuda_home
        assert status.nvcc_path == '/usr/bin/nvcc'
        assert status.cuda_version == '12.0'
        assert isinstance(status.devices, list)

    return True


def test_detect_gpus_via_mock_torch():
    """Detect GPUs using a mocked torch module."""
    print("\n" + "=" * 50)
    print("TEST 5: Detect GPUs via torch")
    print("=" * 50)

    from utils.cuda_utils import CUDAManager

    class _Props:
        total_memory = 8 * 1024 * 1024 * 1024
        major = 8
        minor = 6

    torch = types.SimpleNamespace()
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_properties=lambda idx: _Props(),
        get_device_name=lambda idx: 'FakeTorchGPU',
    )

    with _patched_modules({'torch': torch, 'pynvml': None}):
        cm = CUDAManager()
        devices = cm._detect_gpus_python()

    assert len(devices) == 1
    d0 = devices[0]
    print(f"✓ gpu0: {d0.name} {d0.total_memory_mb}MB cc={d0.compute_capability}")
    assert d0.name == 'FakeTorchGPU'
    assert d0.total_memory_mb == 8192
    assert d0.compute_capability == '8.6'
    return True


def test_detect_gpus_via_mock_nvml():
    """Detect GPUs using a mocked pynvml module."""
    print("\n" + "=" * 50)
    print("TEST 6: Detect GPUs via NVML")
    print("=" * 50)

    from utils.cuda_utils import CUDAManager

    class _Mem:
        total = 4 * 1024 * 1024 * 1024
        free = 3 * 1024 * 1024 * 1024

    class _NVML:
        def nvmlInit(self):
            return None

        def nvmlShutdown(self):
            return None

        def nvmlDeviceGetCount(self):
            return 1

        def nvmlDeviceGetHandleByIndex(self, idx):
            return object()

        def nvmlDeviceGetName(self, handle):
            return b'FakeNVMLGPU'

        def nvmlDeviceGetMemoryInfo(self, handle):
            return _Mem()

    pynvml = _NVML()

    with _patched_modules({'torch': None, 'pynvml': pynvml}):
        cm = CUDAManager()
        devices = cm._detect_gpus_python()

    assert len(devices) == 1
    d0 = devices[0]
    print(f"✓ gpu0: {d0.name} total={d0.total_memory_mb}MB free={d0.free_memory_mb}MB")
    assert d0.name == 'FakeNVMLGPU'
    assert d0.total_memory_mb == 4096
    assert d0.free_memory_mb == 3072
    return True


def test_pywhispercpp_cuda_detection_finds_site_packages_libs():
    """Confirm CUDA-linked libs are detected even if installed at site-packages root."""
    print("\n" + "=" * 50)
    print("TEST 7: pywhispercpp CUDA lib detection")
    print("=" * 50)

    from utils.cuda_utils import CUDAManager

    with tempfile.TemporaryDirectory() as td:
        site_dir = td
        pkg_dir = os.path.join(site_dir, 'pywhispercpp')
        os.makedirs(pkg_dir, exist_ok=True)
        init_path = os.path.join(pkg_dir, '__init__.py')
        with open(init_path, 'w', encoding='utf-8') as f:
            f.write('# dummy\n')

        # Put a native lib at site-packages root (not inside the package)
        lib_path = os.path.join(site_dir, 'libggml-cuda.so')
        with open(lib_path, 'wb') as f:
            f.write(b'\x7fELF')

        pywhispercpp = types.ModuleType('pywhispercpp')
        pywhispercpp.__file__ = init_path
        pywhispercpp_model = types.ModuleType('pywhispercpp.model')
        setattr(pywhispercpp_model, 'Model', type('Model', (), {}))

        def fake_run(cmd, capture_output=True, text=True, timeout=5, **kwargs):
            # Only ldd matters for this test
            if cmd[0] == 'ldd':
                out = "\tlibcudart.so.12 => /usr/lib/x86_64-linux-gnu/libcudart.so.12 (0x0000)\n"
                return types.SimpleNamespace(returncode=0, stdout=out, stderr='')
            return types.SimpleNamespace(returncode=1, stdout='', stderr='')

        with _patched_modules({'pywhispercpp': pywhispercpp, 'pywhispercpp.model': pywhispercpp_model}):
            with mock.patch('utils.cuda_utils.subprocess.run', side_effect=fake_run):
                cm = CUDAManager()
                ok = cm._check_pywhispercpp_cuda()

        print(f"✓ detected: {ok}")
        assert ok is True
        return True


def run_all_tests():
    """Run all CUDA utility tests."""
    print("\n" + "=" * 60)
    print("CUDA UTILS MODULE TEST SUITE")
    print("=" * 60)

    tests = [
        test_cuda_detection_runs,
        test_gpu_memory_check_is_safe,
        test_model_fit_check_returns_message,
        test_detect_cuda_uses_nvidia_smi_without_torch,
        test_detect_gpus_via_mock_torch,
        test_detect_gpus_via_mock_nvml,
        test_pywhispercpp_cuda_detection_finds_site_packages_libs,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
                print(f"\n✅ {test.__name__} PASSED")
            else:
                failed += 1
                print(f"\n❌ {test.__name__} FAILED")
        except Exception as e:
            failed += 1
            print(f"\n❌ {test.__name__} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
