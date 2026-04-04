#!/usr/bin/env python3
import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convierte la primera columna de un archivo TUM de epoch a segundos relativos."
    )
    parser.add_argument("input", type=Path, help="Archivo TUM de entrada")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Archivo de salida (por defecto: <input>_sec.tum)",
    )
    parser.add_argument(
        "--keep-absolute",
        action="store_true",
        help="No resta el primer timestamp; solo reescribe el archivo.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"No existe el archivo: {args.input}")

    out_path = args.output or args.input.with_name(
        f"{args.input.stem}_sec{args.input.suffix}"
    )
    rows = []

    with args.input.open("r", encoding="utf-8") as infile:
        for line in infile:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 8:
                raise ValueError(
                    "Formato TUM inválido. Se esperaban al menos 8 columnas: t x y z qx qy qz qw"
                )
            rows.append(parts)

    if not rows:
        raise ValueError(
            f"El archivo está vacío o no contiene poses válidas: {args.input}"
        )

    first_ts = float(rows[0][0])
    for row in rows:
        if not args.keep_absolute:
            row[0] = f"{float(row[0]) - first_ts:.9f}"
        else:
            row[0] = f"{float(row[0]):.9f}"

    with out_path.open("w", encoding="utf-8") as outfile:
        for row in rows:
            outfile.write(" ".join(row[:8]) + "\n")

    mode = "relativos desde 0" if not args.keep_absolute else "absolutos"
    print(f"Entrada: {args.input}")
    print(f"Salida:  {out_path}")
    print(f"Tiempos: {mode}")


if __name__ == "__main__":
    main()
