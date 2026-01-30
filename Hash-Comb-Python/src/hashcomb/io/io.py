"""I/O helpers for HashComb configs and CSV pipelines."""
from __future__ import annotations
import csv, pickle
from pathlib import Path
from typing import Any, List, Iterable

from ..core.exceptions import ConfigIOError, ConfigNotFoundError, InvalidConfigError, MissingColumnError, CSVFormatError


class PklIO:
    """Persist encoder/decoder tables to disk using pickle."""

    @staticmethod
    def savePickle(path: str | Path, obj: Any) -> None:
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("wb") as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            raise ConfigIOError(str(path), op="write", cause=e) from e

    @staticmethod
    def loadPickle(path: str | Path) -> Any:
        try:
            with Path(path).open("rb") as f:
                return pickle.load(f)
        except FileNotFoundError as e:
            raise ConfigNotFoundError(str(path), cause=e) from e
        except pickle.UnpicklingError as e:
            raise InvalidConfigError(str(path), cause=e) from e
        except Exception as e:
            raise ConfigIOError(str(path), op="read", cause=e) from e

    @staticmethod
    def saveConfig(path: str | Path, config: dict) -> None:
        """Save a HashComb configuration container to disk."""
        if not isinstance(config, dict):
            raise ConfigIOError(str(path), op="write", cause=TypeError("config must be a dict"))
        PklIO.savePickle(path, config)

    @staticmethod
    def loadConfig(path: str | Path) -> dict:
        """Load a configuration container (or legacy hashMap dict) from disk."""
        obj = PklIO.loadPickle(path)
        if isinstance(obj, dict):
            if ("hashMap" in obj) or ("schema" in obj) or ("tree" in obj) or ("params" in obj):
                if "params" not in obj:
                    obj["params"] = {}
                return obj
            # Legacy format: plain hashMap dict
            return {
                "schema": "hashcomb.config.legacy",
                "encoder": None,
                "hashMap": obj,
                "tree": None,
                "params": {},
                "salt": None,
            }
        raise InvalidConfigError(str(path), cause=TypeError("unsupported config format"))

    @staticmethod
    def writeLine(path: str | Path, lines: Iterable[str], append: bool = True) -> None:
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append else "w"
            with p.open(mode, encoding="utf-8", newline="") as f:
                for line in lines:
                    f.write(f"{line}\n")
        except Exception as e:
            raise ConfigIOError(str(path), op="write", cause=e) from e


class CsvIO:
    """CSV utilities for encoding/decoding a value column."""

    @staticmethod
    def sniffDialect(path: str) -> csv.Dialect:
        try:
            with open(path, "r", encoding="utf-8", newline="") as file:
                sample = file.read(2048)
        except FileNotFoundError as e:
            raise CSVFormatError("CSV file not found", ctx={"path": path}, cause=e) from e
        except Exception as e:
            raise CSVFormatError("Unable to read CSV for sniffing", ctx={"path": path}, cause=e) from e

        try:
            return csv.Sniffer().sniff(sample)
        except csv.Error:
            class _Default(csv.Dialect):
                delimiter = ","
                quotechar = '"'
                doublequote = True
                skipinitialspace = False
                lineterminator = "\n"
                quoting = csv.QUOTE_MINIMAL
            return _Default()

    @staticmethod
    def readCsv(path: str, skipHeader: int = 0) -> List[List[str]]:
        dialect = CsvIO.sniffDialect(path)
        rows: List[List[str]] = []
        try:
            with open(path, "r", encoding="utf-8", newline="") as file:
                reader = csv.reader(file, dialect)
                for i, singleRow in enumerate(reader):
                    if i < skipHeader:
                        continue
                    rows.append(singleRow)
        except FileNotFoundError as e:
            raise CSVFormatError("CSV file not found", ctx={"path": path}, cause=e) from e
        except Exception as e:
            raise CSVFormatError("Failed to read CSV", ctx={"path": path}, cause=e) from e
        return rows

    @staticmethod
    def encodeCsv(
        inputCSV: str,
        outputCSV: str,
        encoder,
        valueCol: str = "value",
        hashCol: str = "hash",
    ) -> None:
        dialect = CsvIO.sniffDialect(inputCSV)
        try:
            with open(inputCSV, "r", encoding="utf-8", newline="") as fileIn, \
                 open(outputCSV, "w", encoding="utf-8", newline="") as fileOut:

                reader = csv.DictReader(fileIn, dialect=dialect)
                if not reader.fieldnames:
                    raise CSVFormatError("CSV has no header", ctx={"path": inputCSV})

                fieldNames = list(reader.fieldnames)
                if hashCol not in fieldNames:
                    fieldNames.append(hashCol)

                writer = csv.DictWriter(fileOut, fieldnames=fieldNames, dialect=dialect)
                writer.writeheader()

                for singleRow in reader:
                    try:
                        raw = singleRow[valueCol]
                    except KeyError as e:
                        raise MissingColumnError(valueCol) from e
                    try:
                        v = float(raw)
                    except (TypeError, ValueError) as e:
                        raise CSVFormatError(
                            f"Non-numeric value in column '{valueCol}'",
                            ctx={"row": singleRow},
                            cause=e,
                        ) from e
                    singleRow[hashCol] = encoder.encode(v)
                    writer.writerow(singleRow)
        except FileNotFoundError as e:
            raise CSVFormatError("CSV file not found", ctx={"path": inputCSV}, cause=e) from e
        except CSVFormatError:
            raise
        except Exception as e:
            raise CSVFormatError(
                "Failed to encode CSV",
                ctx={"input": inputCSV, "output": outputCSV},
                cause=e,
            ) from e

    @staticmethod
    def decodeCsv(
        inputCSV: str,
        outputCSV: str,
        decoder,
        hashCol: str = "hash",
        decodedValueCol: str = "decoded_value",
    ) -> None:
        dialect = CsvIO.sniffDialect(inputCSV)
        try:
            with open(inputCSV, "r", encoding="utf-8", newline="") as fileIn, \
                 open(outputCSV, "w", encoding="utf-8", newline="") as fileOut:

                reader = csv.DictReader(fileIn, dialect=dialect)
                if not reader.fieldnames:
                    raise CSVFormatError("CSV has no header", ctx={"path": inputCSV})

                fieldNames = list(reader.fieldnames)
                if decodedValueCol not in fieldNames:
                    fieldNames.append(decodedValueCol)

                writer = csv.DictWriter(fileOut, fieldnames=fieldNames, dialect=dialect)
                writer.writeheader()

                for singleRow in reader:
                    try:
                        token = singleRow[hashCol]
                    except KeyError as e:
                        raise MissingColumnError(hashCol) from e
                    singleRow[decodedValueCol] = decoder.decode(token)
                    writer.writerow(singleRow)
        except FileNotFoundError as e:
            raise CSVFormatError("CSV file not found", ctx={"path": inputCSV}, cause=e) from e
        except CSVFormatError:
            raise
        except Exception as e:
            raise CSVFormatError(
                "Failed to decode CSV",
                ctx={"input": inputCSV, "output": outputCSV},
                cause=e,
            ) from e
