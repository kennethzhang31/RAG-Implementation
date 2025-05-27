import asyncio
import json
import argparse
from docling_parse.pdf_parser import DoclingPdfParser, PdfDocument
from docling_core.types.doc.page import TextCellUnit
from PIL import Image
from pathlib import Path
from typing import List, Optional, Tuple, Any, Union, BinaryIO

class Parser:
    def __init__(self):
        self.parser = DoclingPdfParser()
    
    def load_pdf(self, path_or_stream: Union[str, BinaryIO]) -> PdfDocument:
        """
        Load a PDF file.

        Args:
            path_or_stream (Union[str, BinaryIO]): The path to the PDF file or a binary stream.
        """
        return self.parser.load(path_or_stream=path_or_stream)

    def _input_handler(self, path: Union[str, Path, List[Union[str, Path]]]) -> List[Path]:
        """
        Handle the input path or stream.

        Args:
            path (Union[str, Path, List[Union[str, Path]]]): The path to the PDF file or a binary stream.
        """
        if isinstance(path, (str, Path)):
            path = Path(path)
            if path.is_file() and path.suffix.lower() == ".pdf":
                return [path]
            elif path.is_dir():
                return [p for p in path.rglob("*.pdf")]
            else:
                raise ValueError(f"Invalid input: {path}")
        elif isinstance(path, list):
            return [Path(p) for p in path if Path(p).suffix.lower() == ".pdf"]
        else:
            raise TypeError("Input must be a file path, directory path, or list of file paths.")
        
    def _parse_pdf_file(self, pdf_path: Path) -> Tuple[str, List[Tuple[int, str]]]:
        pdf_doc = self.load_pdf(pdf_path)
        filename = pdf_path.name
        doc_level = []
        for page_no, pred_page in pdf_doc.iterate_pages():
            page_level = []
            for line in pred_page.iterate_cells(unit_type=TextCellUnit.LINE):
                page_level.append(line.text)
            page_text = " ".join(page_level)
            doc_level.append((page_no, page_text))
        return (filename, doc_level)
    
    def extract_text(self, path: Union[str, Path, List[Union[str, Path]]]) -> List[Tuple[str, List[Tuple[int, str]]]]:
        pdf_paths = self._input_handler(path)
        return [self._parse_pdf_file(p) for p in pdf_paths]

    async def extract_text_async(self, path: Union[str, Path, List[Union[str, Path]]]) -> List[Tuple[str, List[Tuple[int, str]]]]:
        pdf_paths = self._input_handler(path)
        tasks = [asyncio.to_thread(self._parse_pdf_file, p) for p in pdf_paths]
        return await asyncio.gather(*tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Docling PDF Parser CLI")
    parser.add_argument("path", type=str, help="Path to a PDF file or directory")
    parser.add_argument("--parallel", action="store_true", help="Use asynchronous parsing")
    parser.add_argument("--output", type=str, help="Output JSON file to save results (optional)")

    args = parser.parse_args()
    p = Parser()

    if args.parallel:
        async def main_async():
            results = await p.extract_text_async(args.path)
            for filename, pages in results:
                print(f"\n📄 {filename}")
                for page_no, text in pages:
                    print(f"\nPage {page_no}:")
                    print(text[:500] + "...\n" + "-"*40)
            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"\n✅ Saved extracted text to {args.output}")

        asyncio.run(main_async())
    else:
        results = p.extract_text(args.path)
        for filename, pages in results:
            print(f"\n📄 {filename}")
            for page_no, text in pages:
                print(f"\nPage {page_no}:")
                print(text[:200] + "...\n" + "-"*40)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n✅ Saved extracted text to {args.output}")