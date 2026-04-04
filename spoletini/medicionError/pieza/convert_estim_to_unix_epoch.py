#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


def infer_unit(ts_value: float) -> str:
    abs_ts = abs(ts_value)
    if abs_ts >= 1e17:
        return "ns"
    if abs_ts >= 1e14:
        return "us"
    if abs_ts >= 1e11:
        return "ms"
    return "s"


def to_seconds_scale(unit: str) -> float:
    scales = {
        "s": 1.0,
        "ms": 1e3,
        "us": 1e6,
        "ns": 1e9,
    }
    return scales[unit]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convierte timestamps de estimación a Unix epoch (formato TUM)."
    )
    parser.add_argument("estim", type=Path, help="Archivo de estimación TUM")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Archivo de salida (por defecto: <estim>_epoch.tum)",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Archivo TUM de referencia para tomar el primer timestamp epoch.",
    )
    parser.add_argument(
        "--epoch-start",
        type=float,
        default=None,
        help="Timestamp Unix epoch inicial en segundos (si no usas --reference).",
    )
    parser.add_argument(
        "--estim-unit",
        choices=["auto", "s", "ms", "us", "ns"],
        default="auto",
        help="Unidad de tiempo de estim (por defecto: auto).",
    )
    parser.add_argument(
        "--reference-unit",
        choices=["auto", "s", "ms", "us", "ns"],
        default="auto",
        help="Unidad de tiempo del archivo de referencia (por defecto: auto).",
    )
    return parser.parse_args()


def load_tum(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 8:
        raise ValueError(
            f"Archivo TUM inválido en {path}. Se esperaban al menos 8 columnas."
        )
    return data


def main() -> None:
    args = parse_args()

    estim_data = load_tum(args.estim)

    if args.output is None:
        out_path = args.estim.with_name(f"{args.estim.stem}_epoch{args.estim.suffix}")
    else:
        out_path = args.output

    if args.reference is None and args.epoch_start is None:
        raise ValueError("Debes pasar --reference o --epoch-start.")

    if args.reference is not None:
        ref_data = load_tum(args.reference)
        ref_first = float(ref_data[0, 0])
        ref_unit = (
            infer_unit(ref_first)
            if args.reference_unit == "auto"
            else args.reference_unit
        )
        epoch_start_s = ref_first / to_seconds_scale(ref_unit)
    else:
        epoch_start_s = float(args.epoch_start)
        ref_unit = "s"

    estim_first = float(estim_data[0, 0])
    estim_unit = (
        infer_unit(estim_first) if args.estim_unit == "auto" else args.estim_unit
    )
    estim_scale = to_seconds_scale(estim_unit)

    # Normaliza estim a tiempo relativo en segundos, anclado al primer frame.
    rel_s = (estim_data[:, 0] - estim_data[0, 0]) / estim_scale
    estim_data[:, 0] = epoch_start_s + rel_s

    np.savetxt(out_path, estim_data, fmt="%.9f")

    print(f"Entrada estim:    {args.estim}")
    print(f"Salida:           {out_path}")
    print(f"Unidad estim:     {estim_unit}")
    if args.reference is not None:
        print(f"Referencia:       {args.reference}")
        print(f"Unidad referencia:{ref_unit}")
    else:
        print("Referencia:       --epoch-start manual")
    print(f"Epoch inicial [s]: {epoch_start_s:.9f}")


if __name__ == "__main__":
    main()
