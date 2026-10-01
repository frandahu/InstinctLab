"""Stream off-screen RGB frames into an H.264 MP4 without a display or GUI."""

from __future__ import annotations

from pathlib import Path


class Mp4Recorder:
    def __init__(self, path, fps):
        try:
            import imageio
            import imageio_ffmpeg

            imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError) as error:
            raise RuntimeError(
                'MP4 encoding requires FFmpeg. Install once in the training environment: '
                'python -m pip install "imageio[ffmpeg]"'
            ) from error
        self.path = Path(path)
        self.fps = fps
        self.frames = 0
        self._shape = None
        self._writer = imageio.get_writer(
            str(self.path), format="FFMPEG", fps=fps, codec="libx264", pixelformat="yuv420p",
            macro_block_size=2, ffmpeg_log_level="error", output_params=["-movflags", "+faststart"],
        )

    def append(self, frame):
        import numpy as np

        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("MP4 frames must be uint8 RGB images with shape (height, width, 3)")
        if frame.shape[0] % 2 or frame.shape[1] % 2:
            raise ValueError("H.264 video width and height must be even")
        if self._shape is not None and frame.shape != self._shape:
            raise ValueError("Video resolution changed during recording")
        self._writer.append_data(np.ascontiguousarray(frame))
        self._shape = frame.shape
        self.frames += 1

    def close(self):
        if self._writer is not None:
            writer, self._writer = self._writer, None
            writer.close()
            if self.frames and (not self.path.is_file() or self.path.stat().st_size == 0):
                raise RuntimeError(f"FFmpeg did not produce a nonempty MP4: {self.path}")


def video_camera_pose(case, lane_y):
    """A fixed side view shows the whole selected staircase and its landing."""
    center_x = (case["lane_min_x_m"] + case["lane_max_x_m"]) / 2
    top = max(case["start_height_m"], case["end_height_m"], *case["surface_heights_m"])
    span = case["lane_max_x_m"] - case["lane_min_x_m"]
    eye = (center_x - 0.35 * span, lane_y - max(6.0, span), top + 3.0)
    target = (center_x, lane_y, top / 2 + 0.5)
    return eye, target


def first_render_frame(env):
    """Warm up the off-screen renderer without advancing or resetting the robot."""
    import numpy as np

    for _ in range(20):
        frame = env.render()
        if frame is not None and np.asarray(frame).size and np.any(frame):
            return frame
    raise RuntimeError(
        "Off-screen renderer returned empty/black frames after warmup. "
        "Check the Isaac Sim RTX renderer/driver startup errors; enable_cameras is already set."
    )
