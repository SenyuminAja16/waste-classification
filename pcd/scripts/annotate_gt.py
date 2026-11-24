"""Gambar mask acuan secara manual dengan poligon untuk foto di pcd/ground_truth/selection.csv."""

import sys

import cv2
import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.config import DATA_DIR
from core.image import load_image, resize_43
from pcd.scripts.select_gt_photos import GT_DIR, SELECTION_CSV

MASK_DIR = GT_DIR / "masks"
SCALE = 3
HELP = (
    "p: poligon | b: kuas | e: penghapus | klik kanan + seret: hapus | [ ]: ukuran kuas | "
    "scroll: zoom | klik tengah + seret: geser | 0: tampilan penuh\n"
    "n: tutup poligon | h: tutup sebagai lubang | u: hapus titik | z: batalkan langkah | r: ulang | "
    "Enter: simpan | s: lewati | q: keluar"
)
MODES = {"p": "poligon", "b": "kuas", "e": "penghapus"}

for keymap in [name for name in plt.rcParams if name.startswith("keymap.")]:
    plt.rcParams[keymap] = []
plt.rcParams["toolbar"] = "None"


def mask_path(object_id: str, view: str):
    return MASK_DIR / f"{object_id}-{view}.png"


class Annotator:
    """Menyimpan titik poligon dan mask yang sedang digambar."""

    def __init__(self, image: np.ndarray, original: np.ndarray | None = None, initial: np.ndarray | None = None):
        """Tampilan diambil dari foto asli supaya tajam. initial berisi mask lama saat mengedit."""
        self.image = image
        height, width = image.shape[:2]
        source = image if original is None else original
        interpolation = cv2.INTER_CUBIC if original is None else cv2.INTER_AREA
        self.display = cv2.resize(source, (width * SCALE, height * SCALE), interpolation=interpolation)
        self.initial = np.zeros(self.display.shape[:2], np.uint8)
        if initial is not None:
            self.initial = cv2.resize(initial, (width * SCALE, height * SCALE), interpolation=cv2.INTER_NEAREST)
        self.reset()

    def reset(self) -> None:
        self.mask = self.initial.copy()
        self.points = []
        self.history = []
        self.mode = "poligon"
        self.brush = 15
        self.cursor = None

    def remember(self) -> None:
        """Simpan mask sebelum diubah, untuk tombol z."""
        self.history.append(self.mask.copy())

    def undo(self) -> None:
        if self.history:
            self.mask = self.history.pop()

    def close_polygon(self, value: int) -> None:
        """Isi poligon: 255 untuk objek, 0 untuk lubang."""
        if len(self.points) >= 3:
            self.remember()
            cv2.fillPoly(self.mask, [np.array(self.points, np.int32)], value)
        self.points = []

    def paint(self, start: tuple, end: tuple, value: int) -> None:
        """Tarik garis dari posisi kursor sebelumnya supaya goresan kuas tidak putus-putus saat mouse digerakkan cepat."""
        cv2.line(self.mask, start, end, value, thickness=2 * self.brush)
        cv2.circle(self.mask, end, self.brush, value, -1)

    def render(self) -> np.ndarray:
        """Buat gambar untuk jendela, dengan mask hijau, poligon merah, dan kuas kuning."""
        view = self.display.copy()
        view[self.mask > 0] = (0.5 * view[self.mask > 0] + (0, 127, 0)).astype(np.uint8)
        contours, _ = cv2.findContours(self.mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(view, contours, -1, (0, 255, 0), 1)
        if self.points:
            cv2.polylines(view, [np.array(self.points, np.int32)], False, (0, 0, 255), 1)
            for point in self.points:
                cv2.circle(view, point, 3, (0, 0, 255), -1)
        if self.cursor and self.mode != "poligon":
            cv2.circle(view, self.cursor, self.brush, (0, 255, 255), 1)
        return cv2.cvtColor(view, cv2.COLOR_BGR2RGB)

    def final_mask(self) -> np.ndarray:
        """Perkecil mask ke ukuran asli. Piksel dihitung objek kalau minimal separuhnya tertutup."""
        height, width = self.image.shape[:2]
        small = cv2.resize(self.mask, (width, height), interpolation=cv2.INTER_AREA)
        return np.where(small >= 128, 255, 0).astype(np.uint8)


def annotate(
    image: np.ndarray, title: str, original: np.ndarray | None = None, initial: np.ndarray | None = None
) -> np.ndarray | str:
    """Buka jendela anotasi untuk satu foto dan kembalikan mask-nya atau 'skip'/'quit'."""
    annotator = Annotator(image, original, initial)
    height, width = annotator.display.shape[:2]
    fig = plt.figure(figsize=(8 * width / max(width, height) + 0.4, 8 * height / max(width, height) + 1))
    ax = fig.add_axes((0.02, 0.1, 0.96, 0.85))
    fig.canvas.manager.set_window_title(title)
    ax.axis("off")
    shown = ax.imshow(annotator.render())
    status = fig.text(0.5, 0.97, "", ha="center", va="top", fontsize=10, weight="bold")
    fig.text(0.5, 0.01, HELP, ha="center", va="bottom", fontsize=7)
    outcome = {}
    stroke = {}
    full_view = {"left": -0.5, "right": width - 0.5, "top": -0.5, "bottom": height - 0.5}
    view = dict(full_view)
    pan = {}

    def refresh() -> None:
        zoom = (full_view["right"] - full_view["left"]) / (view["right"] - view["left"])
        status.set_text(f"{title}  |  mode: {annotator.mode}, kuas {annotator.brush}px, zoom {zoom:.1f}x")
        shown.set_data(annotator.render())
        ax.set_xlim(view["left"], view["right"])
        ax.set_ylim(view["bottom"], view["top"])
        fig.canvas.draw_idle()

    def set_view(left: float, top: float, span_x: float, span_y: float) -> None:
        """Atur area tampilan tanpa keluar dari gambar."""
        left = float(np.clip(left, full_view["left"], full_view["right"] - span_x))
        top = float(np.clip(top, full_view["top"], full_view["bottom"] - span_y))
        view.update(left=left, right=left + span_x, top=top, bottom=top + span_y)

    def on_scroll(event) -> None:
        """Zoom dengan scroll, berpusat di posisi kursor."""
        point = position(event)
        if point is None:
            return
        factor = 0.8 if event.button == "up" else 1.25
        span_x = min((view["right"] - view["left"]) * factor, width)
        span_y = min((view["bottom"] - view["top"]) * factor, height)
        if span_x < 40:
            return
        x, y = point
        left = x - (x - view["left"]) * span_x / (view["right"] - view["left"])
        top = y - (y - view["top"]) * span_y / (view["bottom"] - view["top"])
        set_view(left, top, span_x, span_y)
        refresh()

    def position(event):
        """Ubah posisi klik ke koordinat gambar. Klik di luar gambar ditempel ke tepi."""
        if event.x is None:
            return None
        x, y = ax.transData.inverted().transform((event.x, event.y))
        return (int(np.clip(round(x), 0, width - 1)), int(np.clip(round(y), 0, height - 1)))

    def on_press(event) -> None:
        point = position(event)
        if point is None:
            return
        if event.button == 2:
            pan.update(x=event.x, y=event.y, left=view["left"], top=view["top"])
            return
        erase = event.button == 3 or annotator.mode == "penghapus"
        if event.button == 1 and annotator.mode == "poligon":
            annotator.points.append(point)
        elif event.button in (1, 3) and (erase or annotator.mode == "kuas"):
            annotator.remember()
            stroke.update(value=0 if erase else 255, last=point)
            annotator.paint(point, point, stroke["value"])
        refresh()

    def on_move(event) -> None:
        if pan and event.x is not None:
            pixels_per_unit = ax.bbox.width / (view["right"] - view["left"])
            span_x, span_y = view["right"] - view["left"], view["bottom"] - view["top"]
            set_view(pan["left"] - (event.x - pan["x"]) / pixels_per_unit,
                     pan["top"] + (event.y - pan["y"]) / pixels_per_unit, span_x, span_y)
            refresh()
            return
        point = position(event)
        annotator.cursor = point
        if point and "last" in stroke:
            annotator.paint(stroke["last"], point, stroke["value"])
            stroke["last"] = point
        if annotator.mode != "poligon" or stroke:
            refresh()

    def on_release(event) -> None:
        stroke.clear()
        pan.clear()

    def on_key(event) -> None:
        if event.key == "0":
            view.update(full_view)
        elif event.key in MODES:
            annotator.mode = MODES[event.key]
        elif event.key == "]":
            annotator.brush = min(annotator.brush + 3, 60)
        elif event.key == "[":
            annotator.brush = max(annotator.brush - 3, 3)
        elif event.key == "z":
            annotator.undo()
        elif event.key == "n":
            annotator.close_polygon(255)
        elif event.key == "h":
            annotator.close_polygon(0)
        elif event.key == "u" and annotator.points:
            annotator.points.pop()
        elif event.key == "r":
            annotator.reset()
        elif event.key == "enter":
            annotator.close_polygon(255)
            outcome["result"] = annotator.final_mask()
        elif event.key == "s":
            outcome["result"] = "skip"
        elif event.key == "q":
            outcome["result"] = "quit"
        refresh()
        if "result" in outcome:
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    refresh()
    plt.show()
    return outcome.get("result", "quit")


def main() -> None:
    """Anotasi foto yang belum punya mask atau edit mask tertentu."""
    selection = pd.read_csv(SELECTION_CSV)
    total = len(selection)
    edit = set(sys.argv[1:])
    done = sum(mask_path(photo.object_id, photo.view).exists() for photo in selection.itertuples())
    print(f"{done} sudah dianotasi, {total - done} tersisa" if not edit else f"mode edit: {', '.join(sorted(edit))}")
    print(HELP)
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    for number, photo in enumerate(selection.itertuples(), 1):
        path = mask_path(photo.object_id, photo.view)
        if (path.stem not in edit) if edit else path.exists():
            continue
        initial = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if edit else None
        original = load_image(DATA_DIR / photo.raw_path)
        title = f"foto {number}/{total}, selesai {done}  {photo.object_id} {photo.view} ({photo.label})"
        result = annotate(resize_43(original), title, original, initial)
        if isinstance(result, str):
            if result == "quit":
                break
            continue
        cv2.imwrite(str(path), result)
        done += 0 if edit else 1
        print(f"tersimpan ({done}/{total}): {photo.object_id} {photo.view}, luas {(result > 0).mean():.0%}")


if __name__ == "__main__":
    main()
