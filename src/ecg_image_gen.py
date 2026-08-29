"""
ecg_image_gen.py
=================
Renders a 1D ECG beat/strip as an ECG-style PNG image (grid + waveform trace).

Why you need this: MIT-BIH gives you SIGNALS, but your project's stated input
is an IMAGE. Two uses:
  1. Build a labeled image dataset (image -> PVC/Normal) to train/validate
     the CV+ML stage end-to-end.
  2. Produce the final "highlighted ECG image" output using the same
     rendering code, so highlighting coordinates map cleanly back to pixels.

IMPORTANT (fixed 2026-08-23): the axes are forced to fill 100% of the figure
(fig.add_axes([0,0,1,1]) with no tight_layout / no title-induced margins).
This guarantees that pixel column 0 of the saved PNG corresponds EXACTLY to
signal index 0, and the last pixel column corresponds to the last signal
index -- image_to_signal.py's digitizer does a simple linear
x_pixel / width * out_len mapping, and any whitespace margin around the plot
(e.g. from fig.tight_layout(), which reserves padding even with no title/
ticks/spines) breaks that mapping and corrupts the round-trip badly.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def signal_to_ecg_image(signal, out_path, dpi=100, fig_w=6, fig_h=3,
                         highlight_range=None, title=None, beat_markers=None):
    """
    Render `signal` (1D array) as an ECG strip image on standard red grid.

    highlight_range: optional highlight region(s) to shade red -- used for
                      the final XAI-highlighted output. Accepts either a
                      single (start_idx, end_idx) tuple (single-beat mode)
                      or a list of (start_idx, end_idx) tuples (multi-beat
                      strip mode, one shaded band per beat flagged as PVC).
    beat_markers:     optional list of x-indices to draw small tick marks
                      at (e.g. detected R-peak locations in multi-beat mode),
                      purely a visual aid -- has no effect on digitization.
    """
    n = len(signal)
    x = np.arange(n)

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    # Axes fill the ENTIRE figure -- no margins, no tight_layout padding.
    # This is what keeps the CV round-trip (image_to_signal.py) accurate.
    ax = fig.add_axes([0, 0, 1, 1])

    # classic ECG paper grid
    ax.set_facecolor("#fff5f5")
    ax.set_xticks(np.arange(0, n, max(n // 40, 1)), minor=True)
    ax.grid(which="both", color="#ffb3b3", linewidth=0.5)

    ax.plot(x, signal, color="black", linewidth=1.2)

    if highlight_range is not None:
        ranges = highlight_range if isinstance(highlight_range, list) else [highlight_range]
        for s, e in ranges:
            s, e = max(0, s), min(n, e)
            ax.axvspan(s, e, color="red", alpha=0.25)

    if beat_markers is not None:
        ymin, ymax = np.min(signal), np.max(signal)
        for m in beat_markers:
            ax.plot([m, m], [ymax + 0.05 * (ymax - ymin), ymax + 0.1 * (ymax - ymin)],
                    color="blue", linewidth=1.5, clip_on=False)

    ax.set_xlim(0, n - 1)
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)  # avoid black frame being read as trace

    if title:
        # Drawn as text INSIDE the axes (not ax.set_title(), which would
        # reserve external figure space and break the pixel<->index mapping).
        ax.text(0.02, 0.96, title, transform=ax.transAxes, fontsize=9,
                 va="top", ha="left",
                 bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2))

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def build_image_dataset(X, y, out_dir="data/ecg_images"):
    """Render every beat in X to a PNG. Returns list of (path, label)."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    records = []
    for i, (beat, label) in enumerate(zip(X, y)):
        fname = f"beat_{i:05d}_{'pvc' if label == 1 else 'norm'}.png"
        path = os.path.join(out_dir, fname)
        signal_to_ecg_image(beat, path)
        records.append((path, label))
    return records


def build_multi_beat_strip(beats, out_path, gap_samples=20):
    """
    Concatenate several individual beat arrays end-to-end into ONE signal
    and render it as a single continuous multi-beat ECG strip image --
    this simulates what a real captured rhythm strip looks like (several
    consecutive beats in one image), for testing the multi-beat pipeline.

    beats: list of 1D beat arrays (each same length, e.g. BEAT_LEN=200)
    gap_samples: flat baseline samples inserted between beats so they don't
                 visually run into each other (purely cosmetic).
    Returns: (out_path, concatenated_signal, list of true beat-center
              indices in the concatenated signal -- useful for validating
              R-peak detection against ground truth during testing).
    """
    pieces = []
    centers = []
    cursor = 0
    for i, beat in enumerate(beats):
        b = beat - beat.mean()  # zero-baseline each beat before concatenating
        if i > 0:
            pieces.append(np.zeros(gap_samples))
            cursor += gap_samples
        pieces.append(b)
        centers.append(cursor + int(np.argmax(np.abs(b))))
        cursor += len(b)

    full_signal = np.concatenate(pieces)
    fig_w = max(6, 1.2 * len(beats))  # wider figure for more beats
    signal_to_ecg_image(full_signal, out_path, fig_w=fig_w, fig_h=2.2)
    return out_path, full_signal, centers


if __name__ == "__main__":
    from data_loader import load_dataset_demo
    X, y = load_dataset_demo(n_per_class=2)
    signal_to_ecg_image(X[0], "outputs/test_beat.png",
                         title=f"label={'PVC' if y[0] else 'Normal'}")
    print("wrote outputs/test_beat.png")