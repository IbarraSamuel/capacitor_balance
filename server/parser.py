"""Strict CSV and XLSX parser and validator for capacitor inventory uploads."""

from __future__ import annotations

import csv
import io
import math
import re
import xml.etree.ElementTree as ET  # ruff: ignore[suspicious-xml-etree-import]
import zipfile
from typing import Any

import xlsxwriter

REQUIRED_COLUMNS = ("serial", "capacitance", "deviation")
MIN_FILE_ROWS = 2
MAX_ERRORS_REPORTED = 15
MAX_SERIAL_LEN = 64

_CELL_REF_RE = re.compile(r"^([A-Z]+)(\d+)$")
_NS_MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


class ValidationError(Exception):
    """Raised when an uploaded CSV/XLSX fails schema or datatype validation."""

    def __init__(self, message: str, details: list[str] | None = None) -> None:
        """Create a ValidationError with optional row-level details."""
        super().__init__(message)
        self.message = message
        self.details = details or []


def _col_letters_to_index(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def _extract_xlsx_cell(c_el: ET.Element, shared_strings: list[str]) -> str:
    cell_type = c_el.attrib.get("t", "")
    if cell_type == "s":
        v_el = c_el.find(f"{_NS_MAIN}v")
        if v_el is not None and v_el.text is not None:
            s_idx = int(v_el.text)
            if 0 <= s_idx < len(shared_strings):
                return shared_strings[s_idx].strip()
        return ""
    if cell_type == "inlineStr":
        return "".join(t.text or "" for t in c_el.iter(f"{_NS_MAIN}t")).strip()
    v_el = c_el.find(f"{_NS_MAIN}v")
    return v_el.text.strip() if v_el is not None and v_el.text is not None else ""


def _read_shared_strings(zf: zipfile.ZipFile, names: set[str]) -> list[str]:
    if "xl/sharedStrings.xml" not in names:
        return []
    sst_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))  # ruff: ignore[suspicious-xml-element-tree-usage]
    return [
        "".join(t.text or "" for t in si.iter(f"{_NS_MAIN}t"))
        for si in sst_root.findall(f"{_NS_MAIN}si")
    ]


def _read_xlsx_rows(raw_bytes: bytes) -> list[list[str]]:
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw_bytes))
    except zipfile.BadZipFile as exc:
        msg = "Invalid .xlsx file (corrupted or not an Excel workbook)."
        raise ValidationError(msg) from exc

    with zf:
        names = set(zf.namelist())
        shared_strings = _read_shared_strings(zf, names)
        sheets = sorted(
            n
            for n in names
            if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")
        )
        if not sheets:
            msg = "Excel workbook contains no worksheets."
            raise ValidationError(msg)

        sheet_root = ET.fromstring(zf.read(sheets[0]))  # ruff: ignore[suspicious-xml-element-tree-usage]
        sheet_data = sheet_root.find(f"{_NS_MAIN}sheetData")
        if sheet_data is None:
            return []

        rows: list[list[str]] = []
        for row_el in sheet_data.findall(f"{_NS_MAIN}row"):
            cells: dict[int, str] = {}
            max_col = -1
            for c_el in row_el.findall(f"{_NS_MAIN}c"):
                match = _CELL_REF_RE.match(c_el.attrib.get("r", ""))
                col_idx = (
                    _col_letters_to_index(match.group(1)) if match else max_col + 1
                )
                max_col = max(max_col, col_idx)
                cells[col_idx] = _extract_xlsx_cell(c_el, shared_strings)
            if max_col >= 0:
                row_list = [cells.get(i, "") for i in range(max_col + 1)]
                if any(row_list):
                    rows.append(row_list)
        return rows


def _read_csv_rows(raw_bytes: bytes) -> list[list[str]]:
    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        msg = "CSV file must be UTF-8 encoded."
        raise ValidationError(msg) from exc

    reader = csv.reader(io.StringIO(text))
    return [
        [cell.strip() for cell in row]
        for row in reader
        if any(cell.strip() for cell in row)
    ]


def _parse_strict_int(raw: str) -> int:
    cleaned = raw.strip()
    if not cleaned:
        msg = "empty integer"
        raise ValueError(msg)
    try:
        return int(cleaned)
    except ValueError:
        as_float = float(cleaned)
        if not math.isfinite(as_float) or not as_float.is_integer():
            msg = f"not an integer: {raw}"
            raise ValueError(msg) from None
        return int(as_float)


def _validate_row(
    line_no: int,
    row: list[str],
    col_idx: dict[str, int],
    seen_serials: set[str],
) -> tuple[dict[str, Any] | None, list[str]]:
    row_errors: list[str] = []
    serial = row[col_idx["serial"]].strip() if col_idx["serial"] < len(row) else ""
    cap_raw = (
        row[col_idx["capacitance"]].strip() if col_idx["capacitance"] < len(row) else ""
    )
    dev_raw = (
        row[col_idx["deviation"]].strip() if col_idx["deviation"] < len(row) else ""
    )

    if not serial:
        row_errors.append(f"Row {line_no}: 'serial' must be a non-empty string.")
    elif len(serial) > MAX_SERIAL_LEN:
        row_errors.append(
            f"Row {line_no}: 'serial' exceeds {MAX_SERIAL_LEN} characters."
        )
    elif serial in seen_serials:
        row_errors.append(f"Row {line_no}: duplicate 'serial' '{serial}'.")
    else:
        seen_serials.add(serial)

    cap_val = 0.0
    try:
        cap_val = float(cap_raw)
        if not math.isfinite(cap_val) or cap_val <= 0.0:
            row_errors.append(
                f"Row {line_no}: 'capacitance' must be a positive finite float, "
                f"got '{cap_raw}'."
            )
    except ValueError:
        row_errors.append(
            f"Row {line_no}: 'capacitance' must be a float, got '{cap_raw}'."
        )

    dev_val = 0
    try:
        dev_val = _parse_strict_int(dev_raw)
    except ValueError:
        row_errors.append(
            f"Row {line_no}: 'deviation' must be an integer, got '{dev_raw}'."
        )

    if row_errors:
        return None, row_errors
    return {"serial": serial, "capacitance": cap_val, "deviation": dev_val}, []


def parse_and_validate_upload(filename: str, raw_bytes: bytes) -> list[dict[str, Any]]:
    """Validate an uploaded CSV or XLSX file against the required capacitor schema.

    Returns:
        List of validated capacitor dicts ready for SQLite insertion.

    Raises:
        ValidationError: On unsupported format, missing columns, or bad row datatypes.
    """
    lower = filename.lower().strip()
    if lower.endswith(".csv"):
        rows = _read_csv_rows(raw_bytes)
    elif lower.endswith(".xlsx"):
        rows = _read_xlsx_rows(raw_bytes)
    else:
        msg = "Unsupported file type. Please upload a .csv or .xlsx file."
        raise ValidationError(msg)

    if len(rows) < MIN_FILE_ROWS:
        msg = "File must include a header row and at least one capacitor row."
        raise ValidationError(msg)

    headers = [h.strip().lower() for h in rows[0]]
    missing = [col for col in REQUIRED_COLUMNS if col not in headers]
    if missing:
        msg = (
            f"Missing required column(s): {', '.join(missing)}. "
            f"Required columns are: {', '.join(REQUIRED_COLUMNS)}."
        )
        raise ValidationError(msg, [f"Found columns: {', '.join(headers)}"])

    col_idx = {col: headers.index(col) for col in REQUIRED_COLUMNS}
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    seen_serials: set[str] = set()

    for line_no, row in enumerate(rows[1:], start=2):
        rec, row_errs = _validate_row(line_no, row, col_idx, seen_serials)
        if rec is not None:
            records.append(rec)
        else:
            errors.extend(row_errs)
        if len(errors) >= MAX_ERRORS_REPORTED:
            errors.append(f"Validation stopped after {MAX_ERRORS_REPORTED} errors.")
            break

    if errors:
        msg = f"Validation failed with {len(errors)} error(s)."
        raise ValidationError(msg, errors)
    if not records:
        msg = "No valid capacitor rows found in file."
        raise ValidationError(msg)
    return records


def build_template_csv(sample_records: list[dict[str, Any]]) -> bytes:
    """Generate a ready-to-upload CSV template matching the required schema.

    Returns:
        UTF-8 encoded CSV file bytes.
    """
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(list(REQUIRED_COLUMNS))
    for rec in sample_records:
        writer.writerow([rec["serial"], rec["capacitance"], rec["deviation"]])
    return buf.getvalue().encode("utf-8")


def build_template_xlsx(sample_records: list[dict[str, Any]]) -> bytes:
    """Generate a ready-to-upload Excel (.xlsx) template for the required schema.

    Returns:
        Binary XLSX workbook bytes.
    """
    buf = io.BytesIO()
    workbook = xlsxwriter.Workbook(buf, {"in_memory": True})
    sheet = workbook.add_worksheet("Capacitors")
    for c_idx, col_name in enumerate(REQUIRED_COLUMNS):
        sheet.write(0, c_idx, col_name)
    for r_idx, rec in enumerate(sample_records, start=1):
        sheet.write_string(r_idx, 0, str(rec["serial"]))
        sheet.write_number(r_idx, 1, float(rec["capacitance"]))
        sheet.write_number(r_idx, 2, int(rec["deviation"]))
    workbook.close()
    return buf.getvalue()
