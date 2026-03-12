import os
import sys
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox

SPOLETINI_DIR = os.path.dirname(os.path.abspath(__file__))
MAST3R_DIR = os.path.dirname(SPOLETINI_DIR)
LOGS_DIR = os.path.join(MAST3R_DIR, "logs")
VISUALIZAR = os.path.join(SPOLETINI_DIR, "visualizar.py")

# ── Paleta de colores ─────────────────────────────────────────────────────────
BG = "#1e1e2e"
BG_CARD = "#2a2a3d"
FG = "#cdd6f4"
FG_DIM = "#6c7086"
ACCENT = "#89b4fa"
ACCENT_H = "#b4d0fb"
HOVER_BG = "#313244"
SEL_BG = "#45475a"
BORDER = "#45475a"


def get_subfolders():
    return [
        d
        for d in sorted(os.listdir(LOGS_DIR))
        if os.path.isdir(os.path.join(LOGS_DIR, d))
    ]


def launch(root, folder_name):
    folder = os.path.join(LOGS_DIR, folder_name)
    ply = os.path.join(folder, "data.ply")
    poses = os.path.join(folder, "data.txt")

    missing = [f for f in [ply, poses] if not os.path.isfile(f)]
    if missing:
        messagebox.showerror(
            "Archivos no encontrados",
            "Faltan los siguientes archivos en la carpeta seleccionada:\n"
            + "\n".join(os.path.basename(m) for m in missing),
            parent=root,
        )
        return

    root.destroy()
    subprocess.Popen([sys.executable, VISUALIZAR, ply, poses])


def main():
    root = tk.Tk()
    root.title("MASt3R-SLAM · Seleccionar escena")
    root.resizable(False, False)
    root.configure(bg=BG)

    # ── Centrar ventana ───────────────────────────────────────────────────────
    root.update_idletasks()
    w, h = 420, 480
    x = (root.winfo_screenwidth() - w) // 2
    y = (root.winfo_screenheight() - h) // 2
    root.geometry(f"{w}x{h}+{x}+{y}")

    # ── Título ────────────────────────────────────────────────────────────────
    header = tk.Frame(root, bg=BG)
    header.pack(fill="x", padx=24, pady=(24, 0))

    tk.Label(
        header, text="MASt3R-SLAM", font=("Arial", 18, "bold"), bg=BG, fg=ACCENT
    ).pack(anchor="w")
    tk.Label(
        header,
        text="Selecciona una escena para visualizar",
        font=("Arial", 10),
        bg=BG,
        fg=FG_DIM,
    ).pack(anchor="w", pady=(2, 0))

    ttk.Separator(root, orient="horizontal").pack(fill="x", padx=24, pady=14)

    # ── Lista de carpetas ─────────────────────────────────────────────────────
    folders = get_subfolders()
    if not folders:
        messagebox.showerror("Sin carpetas", f"No hay subcarpetas en:\n{LOGS_DIR}")
        root.destroy()
        return

    list_frame = tk.Frame(
        root, bg=BG_CARD, highlightbackground=BORDER, highlightthickness=1, bd=0
    )
    list_frame.pack(fill="both", expand=True, padx=24)

    scrollbar = tk.Scrollbar(
        list_frame,
        bg=BG_CARD,
        troughcolor=BG_CARD,
        activebackground=ACCENT,
        relief="flat",
        bd=0,
    )
    scrollbar.pack(side="right", fill="y")

    listbox = tk.Listbox(
        list_frame,
        yscrollcommand=scrollbar.set,
        selectmode=tk.SINGLE,
        font=("Arial", 11),
        bg=BG_CARD,
        fg=FG,
        selectbackground=SEL_BG,
        selectforeground=ACCENT,
        activestyle="none",
        highlightthickness=0,
        bd=0,
        relief="flat",
    )
    for f in folders:
        listbox.insert(tk.END, f"  📁  {f}")
    listbox.select_set(0)
    listbox.pack(side="left", fill="both", expand=True, padx=4, pady=4)
    scrollbar.config(command=listbox.yview)

    # ── Hover effect ──────────────────────────────────────────────────────────
    _last_hover = [-1]

    def on_motion(event):
        idx = listbox.nearest(event.y)
        if idx != _last_hover[0]:
            if _last_hover[0] >= 0:
                listbox.itemconfig(_last_hover[0], bg=BG_CARD)
            if idx not in listbox.curselection():
                listbox.itemconfig(idx, bg=HOVER_BG)
            _last_hover[0] = idx

    def on_leave(_):
        if _last_hover[0] >= 0:
            listbox.itemconfig(_last_hover[0], bg=BG_CARD)
            _last_hover[0] = -1

    listbox.bind("<Motion>", on_motion)
    listbox.bind("<Leave>", on_leave)

    # ── Botón ─────────────────────────────────────────────────────────────────
    btn_frame = tk.Frame(root, bg=BG)
    btn_frame.pack(fill="x", padx=24, pady=16)

    btn = tk.Button(
        btn_frame,
        text="Abrir visualizador  →",
        font=("Arial", 11, "bold"),
        bg=ACCENT,
        fg=BG,
        activebackground=ACCENT_H,
        activeforeground=BG,
        relief="flat",
        bd=0,
        cursor="hand2",
        padx=16,
        pady=8,
    )
    btn.pack(fill="x")

    def on_open(_event=None):
        sel = listbox.curselection()
        if not sel:
            messagebox.showwarning(
                "Aviso", "Selecciona una carpeta primero.", parent=root
            )
            return
        folder_name = folders[sel[0]]
        launch(root, folder_name)

    btn.config(command=on_open)
    listbox.bind("<Double-Button-1>", on_open)
    listbox.bind("<Return>", on_open)

    root.mainloop()


if __name__ == "__main__":
    main()
