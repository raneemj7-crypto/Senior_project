
from pathlib import Path
import tkinter as tk
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "datasets/CAUCAFall_annotation_review"
IMAGES = BASE / "images/train"
LABELS = BASE / "labels/train"

CLASSES = ["fallen", "sitting", "standing"]

class AnnotationApp:
    def __init__(self, root):
        self.root = root
        self.files = sorted(IMAGES.glob("*.jpg"))
        self.index = 0
        self.boxes = []
        self.selected_class = 0
        self.start = None
        self.scale = 1
        self.tk_image = None

        self.canvas = tk.Canvas(root, bg="black")
        self.canvas.pack()

        controls = tk.Frame(root)
        controls.pack(pady=8)

        for i, name in enumerate(CLASSES):
            tk.Button(
                controls, text=name,
                command=lambda c=i: self.set_class(c)
            ).pack(side="left")

        tk.Button(
            controls, text="Previous",
            command=self.previous
        ).pack(side="left", padx=8)

        tk.Button(
            controls, text="Save & Next",
            command=self.next
        ).pack(side="left")

        tk.Button(
            controls, text="Remove Last Box",
            command=self.remove_last
        ).pack(side="left")

        self.canvas.bind("<ButtonPress-1>", self.mouse_down)
        self.canvas.bind("<ButtonRelease-1>", self.mouse_up)

        self.show_image()

    def set_class(self, class_id):
        self.selected_class = class_id

    def show_image(self):
        if not self.files:
            self.root.title("No images found")
            return

        path = self.files[self.index]
        image = Image.open(path).convert("RGB")
        self.original_size = image.size

        image.thumbnail((1050, 700))
        self.display_size = image.size
        self.scale_x = self.original_size[0] / image.width
        self.scale_y = self.original_size[1] / image.height

        self.tk_image = ImageTk.PhotoImage(image)
        self.canvas.config(
            width=image.width,
            height=image.height
        )
        self.canvas.delete("all")
        self.canvas.create_image(
            0, 0, image=self.tk_image, anchor="nw"
        )

        self.boxes = []
        label_path = LABELS / f"{path.stem}.txt"

        if label_path.exists():
            for line in label_path.read_text().splitlines():
                values = line.split()
                if len(values) != 5:
                    continue

                c = int(values[0])
                x, y, w, h = map(float, values[1:])

                x1 = (x - w / 2) * image.width
                y1 = (y - h / 2) * image.height
                x2 = (x + w / 2) * image.width
                y2 = (y + h / 2) * image.height

                self.boxes.append((c, x1, y1, x2, y2))

        self.draw_boxes()
        self.root.title(
            f"{self.index + 1}/{len(self.files)} - {path.name}"
        )

    def draw_boxes(self):
        self.canvas.delete("box")
        colors = ["red", "yellow", "lime"]

        for c, x1, y1, x2, y2 in self.boxes:
            self.canvas.create_rectangle(
                x1, y1, x2, y2,
                outline=colors[c],
                width=2,
                tags="box"
            )
            self.canvas.create_text(
                x1 + 4, y1 + 4,
                text=CLASSES[c],
                fill=colors[c],
                anchor="nw",
                tags="box"
            )

    def mouse_down(self, event):
        self.start = (event.x, event.y)

    def mouse_up(self, event):
        if self.start is None:
            return

        x1, y1 = self.start
        x2, y2 = event.x, event.y

        if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
            self.boxes.append((
                self.selected_class,
                min(x1, x2),
                min(y1, y2),
                max(x1, x2),
                max(y1, y2)
            ))
            self.draw_boxes()

        self.start = None

    def save(self):
        path = self.files[self.index]
        label_path = LABELS / f"{path.stem}.txt"

        width, height = self.display_size

        lines = []
        for c, x1, y1, x2, y2 in self.boxes:
            x = ((x1 + x2) / 2) / width
            y = ((y1 + y2) / 2) / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height

            lines.append(
                f"{c} {x:.6f} {y:.6f} "
                f"{w:.6f} {h:.6f}"
            )

        label_path.write_text("\n".join(lines))

    def next(self):
        self.save()
        if self.index < len(self.files) - 1:
            self.index += 1
            self.show_image()

    def previous(self):
        self.save()
        if self.index > 0:
            self.index -= 1
            self.show_image()

    def remove_last(self):
        if self.boxes:
            self.boxes.pop()
            self.draw_boxes()

if __name__ == "__main__":
    root = tk.Tk()
    app = AnnotationApp(root)
    root.mainloop()
