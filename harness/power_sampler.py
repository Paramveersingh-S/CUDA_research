import pynvml, threading, time, numpy as np

class PowerSampler:
    """Continuously samples GPU power draw (mW) on a background thread and
    integrates it to joules over the sampled window. Use as a context manager
    around a benchmarking loop."""
    def __init__(self, device_index=0, sample_hz=100):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self.interval = 1.0 / sample_hz
        self._stop = threading.Event()
        self.samples = []  # (timestamp, watts)

    def _run(self):
        while not self._stop.is_set():
            t = time.perf_counter()
            mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)  # milliwatts
            self.samples.append((t, mw / 1000.0))
            time.sleep(self.interval)

    def __enter__(self):
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    def energy_joules(self):
        """Trapezoidal integration of watts over time -> joules."""
        if len(self.samples) < 2:
            return float("nan")
        arr = np.array(self.samples)
        t, w = arr[:, 0], arr[:, 1]
        return float(np.trapz(w, t))

    def mean_watts(self):
        return float(np.mean([w for _, w in self.samples])) if self.samples else float("nan")
