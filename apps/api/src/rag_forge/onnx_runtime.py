"""The single import point for onnxruntime, so process-wide settings apply to every model.

onnxruntime's telemetry is disabled. A local research platform has no reason to send usage
events, and the uploader's background thread can still be handling a response while the
process exits, which aborts it (`recursive_mutex lock failed`, seen on macOS after test runs
that create many sessions).
"""

import onnxruntime as ort

ort.disable_telemetry_events()

__all__ = ["ort"]
