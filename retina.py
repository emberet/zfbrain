"""Retina for the larval zebrafish: a lattice of luminance + optic-flow channels.

The larva's optomotor response is driven by whole-field visual motion, with the
lower posterior field dominant (simZFish, Liu et al. 2025, showed exactly why).
So we read a screenshot through a coarse grid and produce:

  * a luminance field   (R x C)   -> retina / inner retina drive
  * 4 motion channels   (R x C)   -> DSGC direction channels
      motion energy via discrete shifts, a Hassenstein-Reichardt-style
      approximation - the same trick real DSGCs implement.

Usage from the roamer:
    ret = Retina(rows=12, cols=18)
    while loop:
        frame = np.asarray(page.screenshot(..., scale))  # or PIL image
        out = ret.step(frame)      # out["luminance"], out["motion_*"]
        sim.set_drive(retina_idx, out["luminance"].reshape(-1) * GAIN)
        for name, grid in out["motion"].items():
            sim.set_drive(dsgc_idx[name], grid.reshape(-1) * GAIN)
"""

import numpy as np


class Retina:
    def __init__(self, rows=12, cols=18, gain=100.0, flow_shift=3, motion_max=25.0):
        self.rows = rows
        self.cols = cols
        self.gain = gain
        self.shift = flow_shift
        self.motion_max = motion_max  # Hz cap on a DSGC channel: a scroll is not a flash
        # lower-posterior bias: OMR weights the lower posterior field hardest
        yy, xx = np.mgrid[0:rows, 0:cols]
        lower = (yy + 0.5) / rows                       # 0 top -> 1 bottom
        post = ((xx + 0.5) / cols) * 0.5 + 0.5          # slight lateral bias
        self.field_bias = 0.35 + 0.65 * lower * post
        self._prev = None

    def _downsample(self, frame):
        if isinstance(frame, np.ndarray) is False:
            frame = np.asarray(frame)
        if frame.ndim == 3:
            frame = frame.mean(axis=2)
        if frame.shape != (self.rows, self.cols):
            r, c = frame.shape[:2]
            yr = np.linspace(0, r - 1, self.rows).astype(int)
            xc = np.linspace(0, c - 1, self.cols).astype(int)
            frame = frame[np.ix_(yr, xc)]
        return frame.astype(np.float32)

    @staticmethod
    def _motion(curr, prev, shift):
        """Motion energy for one axis direction via displaced-frame difference."""
        out = np.zeros_like(curr)
        if shift <= 0:
            return out
        if curr.shape[1] > 2 * shift:
            out[:, shift:] = np.abs(curr[:, shift:] - prev[:, :-shift])
        return out

    def _flow(self, curr):
        prev = self._prev if self._prev is not None else curr
        s = self.shift
        # each pair detects motion one way; opposite pair is the other direction
        left = self._motion(curr, prev, s)      # stuff moving right -> left shift
        right = self._motion(prev, curr, s) if prev is not curr else np.zeros_like(curr)
        up = np.zeros_like(curr)
        dn = np.zeros_like(curr)
        if curr.shape[0] > 2 * s:
            up[s:, :] = np.abs(curr[s:, :] - prev[:-s, :])
            dn[:-s, :] = np.abs(curr[:-s, :] - prev[s:, :])
        self._prev = curr.copy()
        return {"up": up, "down": dn, "left": left, "right": right}

    def step(self, frame):
        """Returns dict: 'luminance' (R,C) and 'motion' {up,down,left,right}."""
        curr = self._downsample(frame)
        lum = (curr / 255.0) * self.field_bias if curr.max() > 1.0 else curr * self.field_bias
        motion = {k: self.field_bias * g for k, g in self._flow(curr).items()}
        return {"luminance": lum, "motion": motion}

    def rates(self, out, motion_gain=1.0):
        """Flattened per-channel Hz ready for set_drive() on 'retina'/'dsgc_*'."""
        lum = (out["luminance"].reshape(-1) * self.gain).astype(np.float64)
        motion = {
            k: np.clip(g.reshape(-1) * self.gain * motion_gain, 0.0, self.motion_max).astype(np.float64)
            for k, g in out["motion"].items()
        }
        return {"retina": lum, "dsgc_up": motion["up"], "dsgc_down": motion["down"],
                "dsgc_left": motion["left"], "dsgc_right": motion["right"]}