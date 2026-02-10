"""
Audio Capture Module
Handles microphone input and Voice Activity Detection (VAD)
"""

import logging
import threading
import queue
import time
from typing import Optional, Callable, List
from dataclasses import dataclass
from enum import Enum, auto
import numpy as np

from core.recording_spool import FinalizedRecording

logger = logging.getLogger(__name__)

# Try different VAD implementations
try:
    import torch
    import torchaudio
    SILERO_AVAILABLE = True
    logger.info("Silero VAD available")
except ImportError:
    SILERO_AVAILABLE = False
    logger.warning("Silero VAD not available, using WebRTC VAD")

try:
    import webrtcvad
    WEBRTC_AVAILABLE = True
    logger.info("WebRTC VAD available")
except ImportError:
    WEBRTC_AVAILABLE = False
    logger.warning("WebRTC VAD not available")

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
    logger.info("PyAudio available")
except ImportError:
    PYAUDIO_AVAILABLE = False
    logger.warning("PyAudio not available, trying sounddevice...")

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
    logger.info("sounddevice available")
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    logger.error("No audio backend available! Install PyAudio or sounddevice.")


class RecordingState(Enum):
    """Recording state machine states."""
    IDLE = auto()
    LISTENING = auto()
    RECORDING = auto()
    PROCESSING = auto()


@dataclass
class AudioChunk:
    """Represents a chunk of audio data."""
    data: np.ndarray
    timestamp: float
    is_speech: bool = False
    sample_rate: int = 16000


class BaseVAD:
    """Base class for Voice Activity Detection."""
    
    def __init__(self, sample_rate: int = 16000, aggressiveness: int = 2):
        self.sample_rate = sample_rate
        self.aggressiveness = aggressiveness
        logger.info(f"Initialized {self.__class__.__name__} (aggressiveness={aggressiveness})")
    
    def is_speech(self, audio_frame: bytes) -> bool:
        """Check if audio frame contains speech."""
        raise NotImplementedError
    
    def reset(self):
        """Reset VAD state."""
        pass


class EnergyVAD(BaseVAD):
    """Energy/RMS-based VAD fallback.

    Pure numpy implementation intended as a safe fallback when WebRTC VAD wheels
    are not available (common on some ARM setups).

    Uses simple RMS thresholding with hysteresis and hangover frames.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        aggressiveness: int = 2,
        *,
        start_threshold: Optional[float] = None,
        stop_threshold: Optional[float] = None,
        hangover_frames: Optional[int] = None,
    ):
        super().__init__(sample_rate, aggressiveness)

        # aggressiveness: higher => stricter thresholds (less speech)
        # RMS is computed on float32 normalized to [-1, 1].
        start_map = {0: 0.010, 1: 0.015, 2: 0.020, 3: 0.030}
        stop_map = {0: 0.008, 1: 0.012, 2: 0.016, 3: 0.024}
        hang_map = {0: 10, 1: 8, 2: 6, 3: 5}

        self.start_threshold = float(start_threshold if start_threshold is not None else start_map.get(aggressiveness, 0.020))
        self.stop_threshold = float(stop_threshold if stop_threshold is not None else stop_map.get(aggressiveness, 0.016))
        self.hangover_frames = int(hangover_frames if hangover_frames is not None else hang_map.get(aggressiveness, 6))

        self._in_speech = False
        self._hang = 0
        logger.info(
            "EnergyVAD initialized: start_threshold=%.4f stop_threshold=%.4f hangover_frames=%d",
            self.start_threshold,
            self.stop_threshold,
            self.hangover_frames,
        )

    def is_speech(self, audio_frame: bytes) -> bool:
        try:
            x = np.frombuffer(audio_frame, dtype=np.int16)
            if x.size == 0:
                return False
            xf = x.astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(xf * xf)))

            if not self._in_speech:
                if rms >= self.start_threshold:
                    self._in_speech = True
                    self._hang = self.hangover_frames
                    return True
                return False

            # Currently in speech
            if rms >= self.stop_threshold:
                self._hang = self.hangover_frames
                return True

            # Below stop threshold: hangover
            if self._hang > 0:
                self._hang -= 1
                return True

            self._in_speech = False
            return False
        except Exception as e:
            logger.error(f"EnergyVAD error: {e}")
            return False

    def reset(self):
        self._in_speech = False
        self._hang = 0


class WebRTCVAD(BaseVAD):
    """WebRTC-based Voice Activity Detection."""
    
    def __init__(self, sample_rate: int = 16000, aggressiveness: int = 2):
        super().__init__(sample_rate, aggressiveness)
        
        if not WEBRTC_AVAILABLE:
            raise ImportError("WebRTC VAD not installed. Run: pip install webrtcvad")
        
        self.vad = webrtcvad.Vad(aggressiveness)
        
        # WebRTC VAD only supports specific frame durations
        self.frame_duration_ms = 30  # 10, 20, or 30 ms
        self.frame_size = int(sample_rate * self.frame_duration_ms / 1000)
        
        logger.info(f"WebRTC VAD initialized: sample_rate={sample_rate}, frame_duration={self.frame_duration_ms}ms")
    
    def is_speech(self, audio_frame: bytes) -> bool:
        """Check if audio frame contains speech."""
        try:
            return self.vad.is_speech(audio_frame, self.sample_rate)
        except Exception as e:
            logger.error(f"WebRTC VAD error: {e}")
            return False


class SileroVAD(BaseVAD):
    """Silero-based Voice Activity Detection (more accurate but requires torch)."""
    
    def __init__(self, sample_rate: int = 16000, aggressiveness: int = 2):
        super().__init__(sample_rate, aggressiveness)
        
        if not SILERO_AVAILABLE:
            raise ImportError("Silero VAD requires torch. Run: pip install torch torchaudio")
        
        # Load Silero VAD model
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        
        (self.get_speech_timestamps,
         self.save_audio,
         self.read_audio,
         self.VADIterator,
         self.collect_chunks) = utils
        
        # Map aggressiveness to threshold
        thresholds = {0: 0.15, 1: 0.25, 2: 0.35, 3: 0.5}
        self.threshold = thresholds.get(aggressiveness, 0.35)
        
        self.vad_iterator = self.VADIterator(self.model)
        
        logger.info(f"Silero VAD initialized: threshold={self.threshold}")
    
    def is_speech(self, audio_frame: bytes) -> bool:
        """Check if audio frame contains speech."""
        try:
            # Convert bytes to tensor
            audio_tensor = torch.frombuffer(audio_frame, dtype=torch.int16).float() / 32768.0
            
            # Run VAD
            with torch.no_grad():
                speech_prob = self.model(audio_tensor, self.sample_rate).item()
            
            return speech_prob > self.threshold
        except Exception as e:
            logger.error(f"Silero VAD error: {e}")
            return False
    
    def reset(self):
        """Reset VAD iterator state."""
        self.vad_iterator.reset_states()


class AudioCapture:
    """
    Real-time audio capture with Voice Activity Detection.
    
    Emits audio chunks with speech detected for further processing.
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 channels: int = 1,
                 chunk_size: int = 1024,
                 vad_type: str = "webrtc",
                 vad_aggressiveness: int = 2,
                 silence_timeout: float = 1.5,
                 max_recording_duration: float = 30.0,
                 device_index: Optional[int] = None,
                 gain_db: float = 0.0):
        """
        Initialize audio capture.
        
        Args:
            sample_rate: Audio sample rate (16kHz for Whisper)
            channels: Number of channels (1 for mono)
            chunk_size: Frames per buffer
            vad_type: "webrtc" or "silero"
            vad_aggressiveness: 0-3, higher = more aggressive filtering
            silence_timeout: Seconds of silence to end recording
            device_index: Audio input device index (None = default)
            gain_db: Gain in decibels (0 = no change, positive = louder, negative = quieter)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.silence_timeout = silence_timeout
        self.max_recording_duration = max_recording_duration
        self.device_index = device_index
        self.gain_db = gain_db
        
        logger.info("=" * 50)
        logger.info("Initializing AudioCapture")
        logger.info("=" * 50)
        logger.info(f"Sample rate: {sample_rate} Hz")
        logger.info(f"Channels: {channels}")
        logger.info(f"Chunk size: {chunk_size}")
        logger.info(f"VAD type: {vad_type}")
        logger.info(f"Silence timeout: {silence_timeout}s")
        logger.info(f"Max recording duration: {max_recording_duration}s")
        logger.info(f"Device index: {device_index if device_index is not None else 'default'}")
        logger.info(f"Gain: {gain_db} dB")
        
        # Initialize VAD
        try:
            if vad_type == "silero":
                if SILERO_AVAILABLE:
                    self.vad = SileroVAD(sample_rate, vad_aggressiveness)
                elif WEBRTC_AVAILABLE:
                    logger.warning("Silero VAD requested but unavailable; falling back to WebRTC VAD")
                    self.vad = WebRTCVAD(sample_rate, vad_aggressiveness)
                else:
                    logger.warning("Silero VAD requested but unavailable; falling back to EnergyVAD")
                    self.vad = EnergyVAD(sample_rate, vad_aggressiveness)
            elif vad_type == "energy":
                self.vad = EnergyVAD(sample_rate, vad_aggressiveness)
            elif vad_type == "webrtc":
                if WEBRTC_AVAILABLE:
                    self.vad = WebRTCVAD(sample_rate, vad_aggressiveness)
                else:
                    logger.warning("WebRTC VAD requested but unavailable; falling back to EnergyVAD")
                    self.vad = EnergyVAD(sample_rate, vad_aggressiveness)
            else:
                # Auto: prefer WebRTC, else Energy.
                if WEBRTC_AVAILABLE:
                    self.vad = WebRTCVAD(sample_rate, vad_aggressiveness)
                else:
                    self.vad = EnergyVAD(sample_rate, vad_aggressiveness)
        except Exception as e:
            logger.warning(f"Failed to initialize requested VAD ({vad_type}): {e}; capturing without VAD")
            self.vad = None
        
        # State
        self.state = RecordingState.IDLE
        self.audio_buffer: List[AudioChunk] = []
        self._buffered_samples: int = 0
        self.silence_start_time: Optional[float] = None
        
        # Threading
        self._capture_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._audio_queue: queue.Queue[AudioChunk] = queue.Queue(maxsize=100)
        
        # Callbacks
        self.on_audio_ready: Optional[Callable[[np.ndarray], None]] = None
        self.on_recording_ready: Optional[Callable[[FinalizedRecording], None]] = None
        self.on_state_change: Optional[Callable[[RecordingState], None]] = None
        self.on_speech_detected: Optional[Callable[[bool], None]] = None
        
        # PyAudio
        self._pa: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None
        
        # Calculate gain multiplier
        self._gain_multiplier = 10 ** (gain_db / 20.0)
        self._capture_sample_rate = sample_rate
        
        logger.info("AudioCapture initialized successfully")

    @staticmethod
    def _resample_int16(audio_data: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        """Resample mono int16 audio using linear interpolation."""
        if src_rate <= 0 or dst_rate <= 0:
            return audio_data
        if src_rate == dst_rate or audio_data.size == 0:
            return audio_data

        dst_len = int(round((audio_data.size * float(dst_rate)) / float(src_rate)))
        if dst_len <= 0:
            return np.array([], dtype=np.int16)

        src_idx = np.arange(audio_data.size, dtype=np.float32)
        dst_idx = np.linspace(0.0, float(audio_data.size - 1), num=dst_len, dtype=np.float32)
        out = np.interp(dst_idx, src_idx, audio_data.astype(np.float32))
        return out.clip(-32768, 32767).astype(np.int16)

    def _resolve_sounddevice_input(self):
        """Resolve usable sounddevice input device and capture sample rate."""
        all_devices = sd.query_devices()
        input_devices = [(i, d) for i, d in enumerate(all_devices) if d.get('max_input_channels', 0) > 0]
        if not input_devices:
            raise RuntimeError("No input devices available")

        selected_index = None
        selected_info = None

        # Explicit device index wins when valid.
        if self.device_index is not None and int(self.device_index) >= 0:
            idx = int(self.device_index)
            info = sd.query_devices(idx)
            if info.get('max_input_channels', 0) <= 0:
                raise RuntimeError(f"Selected device #{idx} is not an input device")
            selected_index = idx
            selected_info = info
        else:
            # Default input from host API.
            try:
                default_input = int(sd.default.device[0])
            except Exception:
                default_input = -1

            if default_input >= 0:
                try:
                    info = sd.query_devices(default_input)
                    if info.get('max_input_channels', 0) > 0:
                        selected_index = default_input
                        selected_info = info
                except Exception:
                    selected_index = None

            # Fallback to first available input.
            if selected_index is None:
                selected_index, selected_info = input_devices[0]

        target_rate = int(self.sample_rate)
        capture_rate = target_rate
        max_input_channels = int(selected_info.get('max_input_channels', 1) or 1)
        channel_count = max(1, min(int(self.channels), max_input_channels))

        try:
            sd.check_input_settings(device=selected_index, channels=channel_count, samplerate=target_rate, dtype=np.int16)
        except Exception as e:
            dev_rate = int(selected_info.get('default_samplerate', 0) or 0)
            if dev_rate <= 0:
                raise RuntimeError(f"Requested sample rate {target_rate} not supported: {e}")

            sd.check_input_settings(device=selected_index, channels=channel_count, samplerate=dev_rate, dtype=np.int16)
            capture_rate = dev_rate
            logger.warning(
                "Requested sample rate %d Hz not supported by device #%d; using %d Hz and resampling",
                target_rate,
                selected_index,
                capture_rate,
            )

        return selected_index, selected_info, channel_count, capture_rate
    
    def set_gain(self, gain_db: float):
        """Update gain setting (can be called while recording)."""
        self.gain_db = gain_db
        self._gain_multiplier = 10 ** (gain_db / 20.0)
        logger.info(f"Gain updated to {gain_db} dB (multiplier: {self._gain_multiplier:.2f})")
    
    def set_device(self, device_index: Optional[int]):
        """Update device index (only takes effect on next start())."""
        if self.is_recording():
            logger.warning("Cannot change device while recording. Stop first.")
            return False
        self.device_index = device_index
        logger.info(f"Device index set to {device_index if device_index is not None else 'default'}")
        return True
    
    def _set_state(self, new_state: RecordingState):
        """Update recording state and notify listeners."""
        if self.state != new_state:
            old_state = self.state
            self.state = new_state
            logger.info(f"State changed: {old_state.name} -> {new_state.name}")
            
            if self.on_state_change:
                try:
                    self.on_state_change(new_state)
                except Exception as e:
                    logger.error(f"State change callback error: {e}")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for real-time audio processing."""
        try:
            # Convert to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            
            # Create audio chunk
            chunk = AudioChunk(
                data=audio_data,
                timestamp=time.time(),
                sample_rate=self.sample_rate
            )
            
            # VAD processing if enabled
            if self.vad:
                # WebRTC requires specific frame sizes
                if isinstance(self.vad, WebRTCVAD):
                    frame_bytes = audio_data.tobytes()
                    chunk.is_speech = self.vad.is_speech(frame_bytes)
                else:
                    frame_bytes = audio_data.tobytes()
                    chunk.is_speech = self.vad.is_speech(frame_bytes)
                
                self._process_vad(chunk)
            else:
                # No VAD, just queue everything
                chunk.is_speech = True
                self._queue_chunk(chunk)
            
            return (in_data, pyaudio.paContinue)
            
        except Exception as e:
            logger.error(f"Audio callback error: {e}")
            return (in_data, pyaudio.paContinue)
    
    def _process_vad(self, chunk: AudioChunk):
        """Process VAD result and manage recording state."""
        current_time = time.time()
        
        if chunk.is_speech:
            # Speech detected
            if self.state == RecordingState.LISTENING:
                logger.info("Speech detected! Starting recording...")
                self._set_state(RecordingState.RECORDING)
                self.silence_start_time = None
            
            if self.state == RecordingState.RECORDING:
                self.audio_buffer.append(chunk)
                self._buffered_samples += int(chunk.data.size)
                self.silence_start_time = None

                # Hard cap segment length during continuous speech
                if self.max_recording_duration and self.max_recording_duration > 0:
                    buffered_s = self._buffered_samples / float(self.sample_rate)
                    if buffered_s >= float(self.max_recording_duration):
                        logger.info(
                            f"Max segment duration reached ({buffered_s:.1f}s), finalizing and continuing recording"
                        )
                        self._finalize_audio(resume_state=RecordingState.RECORDING)
            
            # Notify speech detection
            if self.on_speech_detected:
                try:
                    self.on_speech_detected(True)
                except Exception as e:
                    logger.error(f"Speech detected callback error: {e}")
        
        else:
            # Silence
            if self.state == RecordingState.RECORDING:
                # Still record a bit of silence for context
                self.audio_buffer.append(chunk)
                self._buffered_samples += int(chunk.data.size)
                
                # Check if silence timeout reached
                if self.silence_start_time is None:
                    self.silence_start_time = current_time
                    logger.debug("Starting silence timer...")
                
                silence_duration = current_time - self.silence_start_time
                if silence_duration >= self.silence_timeout:
                    logger.info(f"Silence timeout reached ({silence_duration:.1f}s), finalizing audio...")
                    self._finalize_audio(resume_state=RecordingState.LISTENING)
                    self._set_state(RecordingState.LISTENING)
                    self.silence_start_time = None
            
            # Notify silence
            if self.on_speech_detected:
                try:
                    self.on_speech_detected(False)
                except Exception as e:
                    logger.error(f"Speech detected callback error: {e}")
    
    def _queue_chunk(self, chunk: AudioChunk):
        """Queue audio chunk for processing."""
        try:
            self._audio_queue.put_nowait(chunk)
        except queue.Full:
            logger.warning("Audio queue full, dropping chunk")
    
    def _finalize_audio(self, *, resume_state: RecordingState = RecordingState.LISTENING):
        """Finalize current audio buffer and emit.

        resume_state controls what state we return to after emitting.
        - LISTENING: typical end-of-utterance finalize
        - RECORDING: forced segment split during continuous speech
        """
        if not self.audio_buffer:
            return
        
        try:
            # Concatenate all audio chunks
            chunk_count = len(self.audio_buffer)
            audio_data = np.concatenate([chunk.data for chunk in self.audio_buffer])
            duration = len(audio_data) / self.sample_rate

            # Timestamp bounds
            start_ts = float(self.audio_buffer[0].timestamp) if self.audio_buffer else time.time() - duration
            last = self.audio_buffer[-1]
            end_ts = float(last.timestamp) + (len(last.data) / float(self.sample_rate))
            
            logger.info(f"Finalizing audio: {len(self.audio_buffer)} chunks, {duration:.1f}s")
            
            # Clear buffer
            self.audio_buffer = []
            self._buffered_samples = 0
            
            # Emit recording (preferred)
            recording = FinalizedRecording(
                audio_int16=audio_data,
                sample_rate=self.sample_rate,
                channels=self.channels,
                start_ts=start_ts,
                end_ts=end_ts,
                chunks=chunk_count,
            )

            prev_state = self.state
            self._set_state(RecordingState.PROCESSING)

            try:
                if self.on_recording_ready:
                    self.on_recording_ready(recording)
                if self.on_audio_ready:
                    # Backward-compatible hook
                    self.on_audio_ready(audio_data)
            except Exception as e:
                logger.error(f"Audio ready callback error: {e}")
            finally:
                # Restore requested state (or previous state if stopping)
                if self._stop_event.is_set():
                    self._set_state(prev_state)
                else:
                    self._set_state(resume_state)
            
        except Exception as e:
            logger.error(f"Error finalizing audio: {e}")
            self.audio_buffer = []
            self._buffered_samples = 0
    
    def start(self) -> bool:
        """Start audio capture."""
        if self._capture_thread and self._capture_thread.is_alive():
            logger.warning("Audio capture already running")
            return True
        
        try:
            logger.info("Starting audio capture...")
            
            # Try PyAudio first, fall back to sounddevice
            if PYAUDIO_AVAILABLE:
                return self._start_pyaudio()
            elif SOUNDDEVICE_AVAILABLE:
                return self._start_sounddevice()
            else:
                logger.error("Cannot start - No audio backend available")
                return False
            
        except Exception as e:
            logger.error(f"Failed to start audio capture: {e}")
            self.stop()
            return False
    
    def _start_pyaudio(self) -> bool:
        """Start audio capture using PyAudio."""
        logger.info("Using PyAudio backend")
        
        # Initialize PyAudio
        self._pa = pyaudio.PyAudio()
        
        # Get device info if specified
        if self.device_index is not None:
            try:
                device_info = self._pa.get_device_info_by_index(self.device_index)
                logger.info(f"Using device #{self.device_index}: {device_info['name']}")
            except Exception as e:
                logger.warning(f"Could not get device #{self.device_index} info: {e}")
        
        # Open stream with device if specified
        stream_kwargs = {
            'format': pyaudio.paInt16,
            'channels': self.channels,
            'rate': self.sample_rate,
            'input': True,
            'frames_per_buffer': self.chunk_size,
        }
        
        if self.device_index is not None:
            stream_kwargs['input_device_index'] = self.device_index
        
        self._stream = self._pa.open(**stream_kwargs)
        
        # Start stream
        self._stream.start_stream()
        
        # Set state
        self._set_state(RecordingState.LISTENING)
        self._stop_event.clear()
        
        logger.info(f"Audio capture started successfully (PyAudio) with {self.gain_db} dB gain")
        return True
    
    def _start_sounddevice(self) -> bool:
        """Start audio capture using sounddevice."""
        logger.info("Using sounddevice backend")
        
        # Frame accumulator for VAD (WebRTC needs specific frame sizes)
        self._frame_buffer = np.array([], dtype=np.int16)
        self._vad_frame_size = int(self.sample_rate * 30 / 1000)  # 30ms frames
        
        selected_device, device_info, channel_count, capture_rate = self._resolve_sounddevice_input()
        self._capture_sample_rate = int(capture_rate)
        logger.info(
            "Using device #%d: %s (capture_rate=%dHz, target_rate=%dHz, channels=%d)",
            selected_device,
            device_info.get('name', 'unknown'),
            self._capture_sample_rate,
            self.sample_rate,
            channel_count,
        )
        
        def audio_callback(indata, frames, time_info, status):
            """Callback for sounddevice."""
            if status:
                logger.warning(f"Sounddevice status: {status}")
            
            # sounddevice is configured to provide int16 directly (dtype=np.int16)
            audio_data = indata.flatten()
            
            # Ensure we have the right dtype
            if audio_data.dtype != np.int16:
                logger.warning(f"Unexpected audio dtype: {audio_data.dtype}, converting to int16")
                if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                    audio_data = (audio_data * 32767).clip(-32768, 32767).astype(np.int16)
                else:
                    audio_data = audio_data.astype(np.int16)
            
            if self._capture_sample_rate != self.sample_rate:
                audio_data = self._resample_int16(audio_data, self._capture_sample_rate, self.sample_rate)

            # Apply gain if set
            if self._gain_multiplier != 1.0:
                # Convert to float, apply gain, clip, convert back to int16
                audio_float = audio_data.astype(np.float32) * self._gain_multiplier
                audio_data = audio_float.clip(-32768, 32767).astype(np.int16)
            
            # Buffer for VAD frame size alignment
            self._frame_buffer = np.concatenate([self._frame_buffer, audio_data])
            
            # Process complete frames
            while len(self._frame_buffer) >= self._vad_frame_size:
                frame = self._frame_buffer[:self._vad_frame_size]
                self._frame_buffer = self._frame_buffer[self._vad_frame_size:]
                self._process_audio_chunk(frame)
        
        # Start input stream with device if specified
        stream_kwargs = {
            'samplerate': self._capture_sample_rate,
            'channels': channel_count,
            'dtype': np.int16,
            'blocksize': self.chunk_size,
            'callback': audio_callback,
            'device': selected_device,
        }

        self._stream = sd.InputStream(**stream_kwargs)
        
        self._stream.start()
        
        # Set state
        self._set_state(RecordingState.LISTENING)
        self._stop_event.clear()
        
        logger.info(f"Audio capture started successfully (sounddevice) with {self.gain_db} dB gain")
        return True
    
    def _process_audio_chunk(self, audio_data: np.ndarray):
        """Process a chunk of audio data."""
        try:
            # Create audio chunk
            chunk = AudioChunk(
                data=audio_data,
                timestamp=time.time(),
                sample_rate=self.sample_rate
            )
            
            # VAD processing if enabled
            if self.vad and isinstance(self.vad, WebRTCVAD):
                # WebRTC VAD needs exact frame sizes
                try:
                    frame_bytes = audio_data.tobytes()
                    if len(frame_bytes) == self._vad_frame_size * 2:  # 16-bit = 2 bytes
                        chunk.is_speech = self.vad.is_speech(frame_bytes)
                    else:
                        # Skip VAD for wrong-sized frames
                        chunk.is_speech = True
                except Exception as e:
                    # VAD error, assume speech
                    chunk.is_speech = True
                self._process_vad(chunk)
            elif self.vad:
                # Other VAD types
                try:
                    frame_bytes = audio_data.tobytes()
                    chunk.is_speech = self.vad.is_speech(frame_bytes)
                except:
                    chunk.is_speech = True
                self._process_vad(chunk)
            else:
                # No VAD, just queue everything
                chunk.is_speech = True
                self.audio_buffer.append(chunk)
                
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
    
    def stop(self):
        """Stop audio capture."""
        logger.info("Stopping audio capture...")
        
        self._stop_event.set()
        
        # Finalize any pending audio
        if self.audio_buffer:
            self._finalize_audio(resume_state=RecordingState.LISTENING)
        
        # Stop and close stream
        if self._stream:
            try:
                if PYAUDIO_AVAILABLE and hasattr(self._stream, 'stop_stream'):
                    # PyAudio stream
                    self._stream.stop_stream()
                    self._stream.close()
                elif SOUNDDEVICE_AVAILABLE:
                    # sounddevice stream
                    self._stream.stop()
                    self._stream.close()
            except Exception as e:
                logger.error(f"Error closing stream: {e}")
            finally:
                self._stream = None
        
        # Terminate PyAudio if used
        if self._pa:
            try:
                self._pa.terminate()
            except Exception as e:
                logger.error(f"Error terminating PyAudio: {e}")
            finally:
                self._pa = None
        
        # Clear frame buffer for sounddevice
        if hasattr(self, '_frame_buffer'):
            self._frame_buffer = np.array([], dtype=np.int16)
        
        self._set_state(RecordingState.IDLE)
        logger.info("Audio capture stopped")
    
    def get_state(self) -> RecordingState:
        """Get current recording state."""
        return self.state
    
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self.state == RecordingState.RECORDING
    
    def get_buffered_duration(self) -> float:
        """Get duration of currently buffered audio."""
        if not self.audio_buffer:
            return 0.0
        total_samples = sum(len(chunk.data) for chunk in self.audio_buffer)
        return total_samples / self.sample_rate
    
    @staticmethod
    def list_devices() -> List[dict]:
        """List available audio input devices."""
        devices = []
        
        # Try sounddevice first (works better on most systems)
        if SOUNDDEVICE_AVAILABLE:
            try:
                all_devices = sd.query_devices()
                default_input = sd.default.device[0]  # [input, output]
                
                for i, device in enumerate(all_devices):
                    if device.get('max_input_channels', 0) > 0:  # Input device
                        devices.append({
                            'index': i,
                            'name': device['name'],
                            'channels': device['max_input_channels'],
                            'sample_rate': int(device.get('default_samplerate', 44100)),
                            'default': (i == default_input)
                        })
                
                if devices:
                    logger.info(f"Found {len(devices)} input devices via sounddevice")
                    return devices
            except Exception as e:
                logger.warning(f"sounddevice device listing failed: {e}")
        
        # Fall back to PyAudio
        if PYAUDIO_AVAILABLE:
            try:
                pa = pyaudio.PyAudio()
                try:
                    default_input_info = pa.get_default_input_device_info()
                    default_index = default_input_info['index']
                except:
                    default_index = None
                
                for i in range(pa.get_device_count()):
                    info = pa.get_device_info_by_index(i)
                    if info['maxInputChannels'] > 0:  # Input device
                        devices.append({
                            'index': i,
                            'name': info['name'],
                            'channels': info['maxInputChannels'],
                            'sample_rate': int(info['defaultSampleRate']),
                            'default': (i == default_index)
                        })
                
                logger.info(f"Found {len(devices)} input devices via PyAudio")
            except Exception as e:
                logger.error(f"Error listing PyAudio devices: {e}")
            finally:
                pa.terminate()
        
        if not devices:
            logger.error("No audio backend available for device listing")
        
        return devices


# Factory function
def create_audio_capture(config=None) -> AudioCapture:
    """Create AudioCapture instance from config or defaults."""
    if config:
        # Get optional settings with defaults
        device_index = getattr(config.audio, 'device_index', None)
        gain_db = getattr(config.audio, 'gain_db', 0.0)
        
        return AudioCapture(
            sample_rate=config.audio.sample_rate,
            channels=config.audio.channels,
            chunk_size=config.audio.chunk_size,
            vad_type="silero" if SILERO_AVAILABLE else "webrtc",
            vad_aggressiveness=config.audio.vad_aggressiveness,
            silence_timeout=config.audio.silence_timeout,
            max_recording_duration=getattr(config.audio, 'max_recording_duration', 30.0),
            device_index=device_index,
            gain_db=gain_db
        )
    else:
        return AudioCapture()
