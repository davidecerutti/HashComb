from __future__ import annotations
import io, os, csv
from typing import Dict, Optional, List, Iterable, Tuple
import pickle

from src.node import Node
from src.tree import Tree

class Utils:

    # --- serializzazione HASH TABLE

    @staticmethod 
    def writeHashTable2File(fileName: str, tree: Tree) -> None:
        table: Dict[str, Node] = {}
        
        def collect(singleNode: Optional[Node]) -> None : 
            if singleNode is None : return
            if singleNode.channel == tree.channels:
                key = singleNode.getValue(True) #isHashed = true
                table[key] = singleNode
                return
            collect(singleNode.left)
            collect(singleNode.right)
        collect(tree.root)

        os.makedirs(os.path.dirname(fileName) or ".", exist_ok=True)
        with open(fileName, "wb") as file:
            pickle.dump(table, file, protocol=pickle.HIGHEST_PROTOCOL)


    @staticmethod
    def readHashTable2File(fileName: str) -> Dict[str, Node]: 
        with open(fileName, "rb") as file:
            table: Dict[str, Node] = pickle.load(file)
        return table

    # --- utilities di CSV

    @staticmethod
    def sniffDialect(path: str) -> csv.Dialect:                         #source stack overflow for a CSV dialect sniffer
        with open(path, "r", encoding="utf-8", newline="") as file:
            sample = file.read(2048)
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
    def openCSV(path: str, skipHeader: int = 0) -> List[List[str]]:
        dialect = Utils.sniffDialect(path)
        rows: List[List[str]] = []
        with open(path, "r", encoding="utf-8", newline="") as file:
            reader = csv.reader(file, dialect)
            for i, singleRow in enumerate(reader):
                if i < skipHeader: continue
                rows.append(singleRow)
        return rows
    
    @staticmethod
    def encode2File(inputCSV: str, outputCSV: str, encoder, valueCol: str = "value", hashCol: str = "hash") -> None:
        dialect = Utils.sniffDialect(inputCSV)
        with open(inputCSV, "r", encoding="utf-8", newline="") as fileIn, open(outputCSV, "w", encoding="utf-8", newline="") as fileOut:
            reader = csv.DictReader(fileIn, dialect=dialect)
            fieldNames = list(reader.fieldnames or [])
            if hashCol not in fieldNames:
                fieldNames.append(hashCol)
            
            writer = csv.DictWriter(fileOut, fieldnames=fieldNames, dialect=dialect)
            writer.writeheader()
            for singleRow in reader:
                try:
                    singleValueCol = float(singleRow[valueCol])
                except Exception as e:
                    raise ValueError(f"missing valueCol in singleRow= {singleRow}") from e
                singleRow[hashCol] = encoder.encode(singleValueCol)
                writer.writerow(singleRow)

    @staticmethod
    def decode2File(inputCSV: str, outputCSV: str, decoder, hashCol: str = "hash", decodedValueCol: str = "decoded_value") -> None:
        dialect = Utils.sniffDialect(inputCSV)
        with open(inputCSV, "r", encoding="utf-8", newline="") as fileIn, open(outputCSV, "w", encoding="utf-8", newline="") as fileOut:
            reader = csv.DictReader(fileIn, dialect=dialect)
            fieldNames = list(reader.fieldnames or [])
            if decodedValueCol not in fieldNames:
                fieldNames.append(decodedValueCol)
            writer = csv.DictWriter(fileOut, fieldnames=fieldNames, dialect=dialect)
            writer.writeheader()
            for singleRow in reader:
                try:
                    singleHashCol = singleRow[hashCol]
                except KeyError as  e:
                    raise ValueError(f"missing hashCol in singleRow= {singleRow}") from e
            singleRow[decodedValueCol] = decoder.decode(singleHashCol)
            writer.writerow(singleRow)

    @staticmethod
    def print2File(path: str, lines: Iterable[str], append: bool = True) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        mode = "a" if append else "w"
        with open(path, mode, encoding="utf-8", newline="") as file:
            for line in lines:
                file.write(f"{line}\n")





