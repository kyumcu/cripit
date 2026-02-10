"""
CUDA Utilities Module
Handles CUDA detection, validation, and GPU resource management for CripIt.

This module provides:
- CUDA availability detection
- GPU validation and memory checking
- Fallback mechanisms with warnings
- Console logging for debugging
"""

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CUDADeviceInfo:
    """Information about a CUDA-capable GPU device."""
    device_id: int
    name: str
    total_memory_mb: int
    free_memory_mb: int
    compute_capability: str
    driver_version: str


@dataclass
class CUDAStatus:
    """Complete CUDA system status."""
    available: bool
    cuda_home: Optional[str]
    nvcc_path: Optional[str]
    cuda_version: Optional[str]
    devices: list
    error_message: Optional[str]
    pywhispercpp_cuda: bool = False


class CUDAManager:
    """
    Manages CUDA detection and GPU resources for whisper.cpp transcription.
    
    Features:
    - Automatic CUDA detection
    - GPU memory monitoring
    - Graceful fallback to CPU
    - Detailed console logging
    """
    
    def __init__(self):
        """Initialize CUDA manager."""
        self._cuda_status: Optional[CUDAStatus] = None
        self._pywhispercpp_cuda_available: Optional[bool] = None
        
        logger.info("=" * 60)
        logger.info("Initializing CUDA Manager")
        logger.info("=" * 60)
    
    def detect_cuda(self, force_check: bool = False) -> CUDAStatus:
        """
        Detect CUDA availability and GPU information.
        
        Args:
            force_check: Force re-check even if already detected
            
        Returns:
            CUDAStatus with complete detection results
        """
        if self._cuda_status is not None and not force_check:
            logger.debug("Returning cached CUDA status")
            return self._cuda_status
        
        logger.info("Detecting CUDA environment...")
        
        status = CUDAStatus(
            available=False,
            cuda_home=None,
            nvcc_path=None,
            cuda_version=None,
            devices=[],
            error_message=None,
            pywhispercpp_cuda=False
        )
        
        # Step 1: Check CUDA_HOME environment variable
        cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
        if cuda_home and not Path(cuda_home).exists():
            logger.warning(f"CUDA_HOME is set but does not exist: {cuda_home}")
            cuda_home = None

        if not cuda_home:
            # Try common CUDA installation paths
            common_paths = [
                "/usr/local/cuda",
                "/usr/local/cuda-12.0",
                "/usr/local/cuda-11.8",
                "/usr/local/cuda-11.7",
                "/opt/cuda",
                "/usr/lib/cuda",
            ]
            for path in common_paths:
                if Path(path).exists():
                    cuda_home = path
                    logger.info(f"Found CUDA installation at: {cuda_home}")
                    break
        
        if cuda_home and Path(cuda_home).exists():
            status.cuda_home = cuda_home
            logger.info(f"CUDA_HOME detected: {cuda_home}")
        else:
            logger.warning("CUDA_HOME not found. Checking alternative methods...")
        
        # Step 2: Check for nvcc compiler
        nvcc_path = None
        if cuda_home:
            nvcc_candidates = [
                Path(cuda_home) / "bin" / "nvcc",
                Path(cuda_home) / "bin" / "nvcc.exe",
            ]
            for candidate in nvcc_candidates:
                if candidate.exists():
                    nvcc_path = str(candidate)
                    break
        
        # Also check PATH
        if not nvcc_path:
            try:
                result = subprocess.run(['which', 'nvcc'], 
                                      capture_output=True, 
                                      text=True, 
                                      timeout=5)
                if result.returncode == 0:
                    nvcc_path = result.stdout.strip()
            except Exception:
                pass
        
        status.nvcc_path = nvcc_path
        if nvcc_path:
            logger.info(f"NVCC found: {nvcc_path}")
            
            # Get CUDA version from nvcc
            try:
                result = subprocess.run([nvcc_path, '--version'], 
                                      capture_output=True, 
                                      text=True, 
                                      timeout=5)
                if result.returncode == 0:
                    # Parse version from output
                    for line in result.stdout.split('\n'):
                        if 'release' in line.lower():
                            # Extract version number
                            import re
                            match = re.search(r'release (\d+\.\d+)', line)
                            if match:
                                status.cuda_version = match.group(1)
                                logger.info(f"CUDA Version: {status.cuda_version}")
                                break
            except Exception as e:
                logger.warning(f"Could not get CUDA version: {e}")
        else:
            logger.warning("NVCC not found in PATH or CUDA_HOME")
        
        # Step 3: Check for CUDA libraries
        if cuda_home:
            lib_paths = [
                Path(cuda_home) / "lib64",
                Path(cuda_home) / "lib",
                Path(cuda_home) / "lib" / "x64",
            ]
            lib_found = any(path.exists() for path in lib_paths)
            if lib_found:
                logger.info("CUDA libraries found")
            else:
                logger.warning("CUDA libraries not found in expected locations")
        
        # Step 4: Try to detect GPUs using Python libraries
        devices = self._detect_gpus_python()
        if devices:
            status.devices = devices
            logger.info(f"Detected {len(devices)} GPU(s):")
            for dev in devices:
                logger.info(f"  GPU {dev.device_id}: {dev.name} ({dev.total_memory_mb}MB)")
        else:
            logger.warning("No GPUs detected via Python libraries")
        
        # Step 5: Determine overall availability
        if nvcc_path and (devices or self._check_nvidia_smi()):
            status.available = True
            logger.info("CUDA is available on this system")
        else:
            status.available = False
            if not nvcc_path:
                status.error_message = "NVCC compiler not found. Install CUDA toolkit."
            elif not devices:
                status.error_message = "No CUDA-capable GPUs detected."
            logger.warning(f"CUDA not available: {status.error_message}")
        
        # Step 6: Check if pywhispercpp has CUDA support
        status.pywhispercpp_cuda = self._check_pywhispercpp_cuda()
        if status.pywhispercpp_cuda:
            logger.info("pywhispercpp compiled with CUDA support")
        else:
            logger.info("pywhispercpp does not have CUDA support (CPU only)")
        
        # Cache the result
        self._cuda_status = status
        
        logger.info("=" * 60)
        logger.info(f"CUDA Detection Complete: {'AVAILABLE' if status.available else 'NOT AVAILABLE'}")
        logger.info("=" * 60)
        
        return status
    
    def _detect_gpus_python(self) -> list:
        """Try to detect GPUs using available Python libraries."""
        devices = []
        
        # Try torch first (most common)
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                for i in range(device_count):
                    props = torch.cuda.get_device_properties(i)
                    devices.append(CUDADeviceInfo(
                        device_id=i,
                        name=torch.cuda.get_device_name(i),
                        total_memory_mb=props.total_memory // (1024 * 1024),
                        free_memory_mb=0,  # Will be updated separately
                        compute_capability=f"{props.major}.{props.minor}",
                        driver_version="unknown"
                    ))
                logger.info(f"Detected {device_count} GPU(s) via PyTorch")
                return devices
        except ImportError:
            logger.debug("PyTorch not available for GPU detection")
        except Exception as e:
            logger.debug(f"PyTorch GPU detection failed: {e}")
        
        # Try pynvml (nvidia-ml-py)
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                devices.append(CUDADeviceInfo(
                    device_id=i,
                    name=name.decode('utf-8') if isinstance(name, bytes) else name,
                    total_memory_mb=mem_info.total // (1024 * 1024),
                    free_memory_mb=mem_info.free // (1024 * 1024),
                    compute_capability="unknown",
                    driver_version="unknown"
                ))
            
            pynvml.nvmlShutdown()
            logger.info(f"Detected {device_count} GPU(s) via NVML")
            return devices
        except ImportError:
            logger.debug("NVML not available for GPU detection")
        except Exception as e:
            logger.debug(f"NVML GPU detection failed: {e}")

        # Fallback: use nvidia-smi (does not require Python GPU libs)
        try:
            result = subprocess.run(
                [
                    'nvidia-smi',
                    '--query-gpu=index,name,memory.total,memory.free',
                    '--format=csv,noheader,nounits',
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                lines = [ln.strip() for ln in result.stdout.split('\n') if ln.strip()]
                for ln in lines:
                    parts = [p.strip() for p in ln.split(',')]
                    if len(parts) < 4:
                        continue
                    try:
                        device_id = int(parts[0])
                        name = parts[1]
                        total_mb = int(float(parts[2]))
                        free_mb = int(float(parts[3]))
                    except Exception:
                        continue
                    devices.append(CUDADeviceInfo(
                        device_id=device_id,
                        name=name,
                        total_memory_mb=total_mb,
                        free_memory_mb=free_mb,
                        compute_capability="unknown",
                        driver_version="unknown",
                    ))
                if devices:
                    logger.info(f"Detected {len(devices)} GPU(s) via nvidia-smi")
                    return devices
        except FileNotFoundError:
            logger.debug("nvidia-smi not found")
        except Exception as e:
            logger.debug(f"nvidia-smi GPU detection failed: {e}")
        
        return devices
    
    def _check_nvidia_smi(self) -> bool:
        """Check if nvidia-smi is available and can detect GPUs."""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                  capture_output=True,
                                  text=True,
                                  timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                gpus = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
                logger.info(f"nvidia-smi detected {len(gpus)} GPU(s)")
                return True
        except FileNotFoundError:
            logger.debug("nvidia-smi not found")
        except Exception as e:
            logger.debug(f"nvidia-smi check failed: {e}")
        
        return False
    
    def _check_pywhispercpp_cuda(self) -> bool:
        """Check if pywhispercpp is compiled with CUDA support."""
        try:
            from pywhispercpp.model import Model
            import pywhispercpp

            # Best-effort checks. pywhispercpp does not guarantee an explicit flag,
            # so we attempt to infer CUDA support by inspecting linked libraries.

            # Method 1: Known attribute flags (if present)
            if hasattr(pywhispercpp, 'CUDA_AVAILABLE'):
                try:
                    return bool(getattr(pywhispercpp, 'CUDA_AVAILABLE'))
                except Exception:
                    pass

            if hasattr(Model, 'CUDA_SUPPORTED'):
                try:
                    return bool(getattr(Model, 'CUDA_SUPPORTED'))
                except Exception:
                    pass

            # Method 2: Inspect shared objects and look for CUDA runtime deps
            try:
                pkg_dir = Path(pywhispercpp.__file__).resolve().parent
                # pywhispercpp wheels may install native libs either inside the
                # package directory (e.g. pywhispercpp/_pywhispercpp*.so) or at
                # the site-packages root (e.g. libggml-cuda.so).
                site_dir = pkg_dir.parent

                so_files = list(pkg_dir.glob('**/*.so'))
                so_files += list(site_dir.glob('libggml*.so*'))
                so_files += list(site_dir.glob('libwhisper*.so*'))

                # De-duplicate while preserving order
                seen = set()
                so_files = [p for p in so_files if not (str(p) in seen or seen.add(str(p)))]
                if not so_files:
                    logger.debug("pywhispercpp has no .so files to inspect")
                    return False

                for so_path in so_files:
                    try:
                        result = subprocess.run(
                            ['ldd', str(so_path)],
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        if result.returncode != 0:
                            continue
                        out = (result.stdout or "") + (result.stderr or "")
                        out_lower = out.lower()
                        if (
                            'libcudart' in out_lower or
                            'libcuda' in out_lower or
                            'libcublas' in out_lower or
                            'libcudnn' in out_lower
                        ):
                            logger.info(f"Detected CUDA-linked pywhispercpp binary: {so_path.name}")
                            return True
                    except Exception as e:
                        logger.debug(f"ldd inspection failed for {so_path}: {e}")
                        continue

            except Exception as e:
                logger.debug(f"Shared library inspection failed: {e}")

            logger.debug("Could not confirm pywhispercpp CUDA support")
            return False
            
        except ImportError:
            logger.warning("pywhispercpp not installed, cannot check CUDA support")
            return False
        except Exception as e:
            logger.debug(f"Error checking pywhispercpp CUDA support: {e}")
            return False
    
    def get_gpu_memory_info(self, device_id: int = 0) -> Optional[Tuple[int, int]]:
        """
        Get GPU memory information for a specific device.
        
        Args:
            device_id: GPU device ID (default: 0)
            
        Returns:
            Tuple of (free_memory_mb, total_memory_mb) or None if unavailable
        """
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.set_device(device_id)
                free_memory = torch.cuda.mem_get_info(device_id)[0]
                total_memory = torch.cuda.get_device_properties(device_id).total_memory
                return (free_memory // (1024 * 1024), total_memory // (1024 * 1024))
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"PyTorch memory check failed: {e}")
        
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            pynvml.nvmlShutdown()
            return (mem_info.free // (1024 * 1024), mem_info.total // (1024 * 1024))
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"NVML memory check failed: {e}")

        # Fallback: use nvidia-smi query
        try:
            result = subprocess.run(
                [
                    'nvidia-smi',
                    '-i', str(device_id),
                    '--query-gpu=memory.free,memory.total',
                    '--format=csv,noheader,nounits',
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                # Example: "12345, 24576"
                parts = [p.strip() for p in result.stdout.strip().split(',')]
                if len(parts) >= 2:
                    free_mb = int(float(parts[0]))
                    total_mb = int(float(parts[1]))
                    return (free_mb, total_mb)
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.debug(f"nvidia-smi memory check failed: {e}")
        
        return None
    
    def check_model_fits_gpu(self, model_name: str, device_id: int = 0) -> Tuple[bool, str]:
        """
        Check if a specific whisper model will fit in GPU memory.
        
        Args:
            model_name: Name of the model (e.g., 'large-v3-turbo')
            device_id: GPU device ID to check
            
        Returns:
            Tuple of (fits: bool, message: str)
        """
        # Model memory requirements (approximate, in MB)
        model_memory_requirements = {
            'tiny': 500,
            'tiny.en': 500,
            'base': 800,
            'base.en': 800,
            'small': 1500,
            'small.en': 1500,
            'medium': 4000,
            'medium.en': 4000,
            'large-v1': 8000,
            'large-v2': 8000,
            'large-v3': 8000,
            'large-v3-turbo': 4000,
            'distil-large-v3': 3500,
        }
        
        required_memory = model_memory_requirements.get(model_name, 4000)
        
        memory_info = self.get_gpu_memory_info(device_id)
        if memory_info is None:
            return (False, "Cannot determine GPU memory availability")
        
        free_memory, total_memory = memory_info
        
        # Add 500MB buffer for overhead
        required_with_buffer = required_memory + 500
        
        if free_memory >= required_with_buffer:
            return (True, f"Model '{model_name}' fits in GPU memory ({free_memory}MB free, {required_memory}MB required)")
        else:
            return (False, f"Model '{model_name}' requires {required_memory}MB but only {free_memory}MB available on GPU")
    
    def validate_cuda_setup(self, verbose: bool = True) -> Tuple[bool, str]:
        """
        Complete validation of CUDA setup for transcription.
        
        Args:
            verbose: Print detailed information to console
            
        Returns:
            Tuple of (is_valid: bool, message: str)
        """
        status = self.detect_cuda()
        
        if verbose:
            print("\n" + "=" * 60)
            print("CUDA Setup Validation Report")
            print("=" * 60)
            print(f"CUDA Available: {'YES' if status.available else 'NO'}")
            print(f"CUDA Home: {status.cuda_home or 'Not found'}")
            print(f"CUDA Version: {status.cuda_version or 'Unknown'}")
            print(f"NVCC Path: {status.nvcc_path or 'Not found'}")
            print(f"pywhispercpp CUDA: {'YES' if status.pywhispercpp_cuda else 'NO'}")
            print(f"GPU Devices: {len(status.devices)}")
            
            for dev in status.devices:
                print(f"  [{dev.device_id}] {dev.name} - {dev.total_memory_mb}MB")
            
            if status.error_message:
                print(f"\nError: {status.error_message}")
            print("=" * 60 + "\n")
        
        if not status.available:
            return (False, status.error_message or "CUDA not available")
        
        if not status.pywhispercpp_cuda:
            return (False, "pywhispercpp not compiled with CUDA support. Run build_cuda.sh")
        
        return (True, "CUDA setup is valid and ready for GPU transcription")
    
    def print_setup_instructions(self):
        """Print instructions for setting up CUDA."""
        print("\n" + "=" * 60)
        print("CUDA Setup Instructions")
        print("=" * 60)
        print("""
To enable GPU acceleration for CripIt:

1. Install CUDA Toolkit (11.8 or 12.x recommended):
   - Ubuntu: sudo apt-get install nvidia-cuda-toolkit
   - Or download from: https://developer.nvidia.com/cuda-downloads

2. Set CUDA_HOME environment variable:
   export CUDA_HOME=/usr/local/cuda
   export PATH=$CUDA_HOME/bin:$PATH
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

3. Build pywhispercpp with CUDA support:
   bash build_cuda.sh

4. Verify installation:
   python -c "from utils.cuda_utils import CUDAManager; CUDAManager().validate_cuda_setup()"

For detailed instructions, see CUDA_SETUP.md
""")
        print("=" * 60 + "\n")


# Singleton instance
_cuda_manager: Optional[CUDAManager] = None


def get_cuda_manager() -> CUDAManager:
    """Get singleton CUDA manager instance."""
    global _cuda_manager
    if _cuda_manager is None:
        _cuda_manager = CUDAManager()
    return _cuda_manager


def quick_cuda_check() -> bool:
    """Quick check if CUDA is ready for transcription."""
    manager = get_cuda_manager()
    is_valid, message = manager.validate_cuda_setup(verbose=False)
    return is_valid


def get_gpu_memory_info() -> Optional[Dict[str, int]]:
    """
    Get GPU memory information.
    
    Returns:
        Dict with 'free_mb', 'total_mb', 'used_mb' or None if CUDA unavailable
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        
        torch.cuda.synchronize()
        free_memory = torch.cuda.mem_get_info()[0]
        total_memory = torch.cuda.mem_get_info()[1]
        
        return {
            'free_mb': free_memory // (1024 * 1024),
            'total_mb': total_memory // (1024 * 1024),
            'used_mb': (total_memory - free_memory) // (1024 * 1024)
        }
    except Exception as e:
        logger.debug(f"Could not get GPU memory info: {e}")
        return None


def check_whisperx_compatibility() -> tuple:
    """
    Check if system can run WhisperX comfortably.
    
    Returns:
        Tuple of (compatible: bool, warning: str, recommendation: str)
    """
    warnings = []
    
    # Check GPU memory
    gpu_info = get_gpu_memory_info()
    if gpu_info:
        if gpu_info['free_mb'] < 4096:  # Less than 4GB free
            warnings.append(f"Low GPU memory: {gpu_info['free_mb']}MB free. "
                          f"WhisperX large model requires ~4GB VRAM.")
    else:
        warnings.append("No CUDA GPU detected. WhisperX will run on CPU (slower).")
    
    # Check system RAM (approximate)
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        if ram_gb < 8:
            warnings.append(f"Low system RAM: {ram_gb:.1f}GB. Recommended: 16GB+ for WhisperX.")
    except ImportError:
        pass
    
    if warnings:
        return (
            False,
            "\n".join(warnings),
            "Consider using whisper.cpp for lower resource usage, or use a smaller model like 'base' or 'tiny'."
        )
    
    return (True, "", "")


def estimate_whisperx_memory(model_name: str, compute_type: str) -> int:
    """
    Estimate VRAM requirement for WhisperX model.
    
    Returns:
        Estimated VRAM in MB
    """
    # Rough estimates based on model size
    base_memory = {
        "tiny": 1000,
        "base": 1500,
        "small": 2500,
        "medium": 4000,
        "large-v3": 6000
    }
    
    memory = base_memory.get(model_name, 4000)
    
    # Adjust for compute type
    if compute_type == "int8":
        memory = int(memory * 0.5)
    elif compute_type == "float16":
        memory = int(memory * 0.75)
    # float32 uses full memory
    
    return memory


if __name__ == "__main__":
    # Standalone test
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    manager = CUDAManager()
    is_valid, message = manager.validate_cuda_setup(verbose=True)
    
    if not is_valid:
        manager.print_setup_instructions()
        sys.exit(1)
    else:
        print("✓ CUDA is properly configured!")
        sys.exit(0)
