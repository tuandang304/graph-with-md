import os
import json
from tqdm import tqdm

class BaselineLoader:
    """
    Baseline: Loader that ignores Section/Heading hierarchy and Table JSON structure.
    Flattens all paper data into a single concatenated string (simulating raw PDF scan text).
    """
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = input_dir
        self.output_dir = output_dir

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    def process_file(self, filename: str):
        filepath = os.path.join(self.input_dir, filename)

        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"Starting raw text compression (PDF simulation mode) for {len(data)} papers from {filename}...")
        for paper_id, content in tqdm(data.items()):
            pdf_text = self._simulate_pdf_text(
                title=content.get('title', ''),
                abstract=content.get('abstract', ''),
                full_text=content.get('full_text', []),
                figures=content.get('figures_and_tables', [])
            )

            output_filepath = os.path.join(self.output_dir, f"{paper_id}.txt")
            with open(output_filepath, 'w', encoding='utf-8') as out_f:
                out_f.write(pdf_text)

    def _simulate_pdf_text(self, title: str, abstract: str, full_text: list, figures: list) -> str:
        plain = []

        # Everything flattened like scanned text — no heading hierarchy
        plain.append(title)
        if abstract:
            plain.append(abstract)

        for section in full_text:
            paragraphs = section.get('paragraphs', [])
            for p in paragraphs:
                plain.append(p)

        if figures:
            for fig in figures:
                caption = fig.get('caption', '')
                plain.append(caption)  # Append caption text only, no image file links

        # Join all with single space to simulate continuous text block
        return " ".join(plain)

if __name__ == "__main__":
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    loader = BaselineLoader(
        input_dir=os.path.join(_root, "data", "raw"),
        output_dir=os.path.join(_root, "data", "baseline_parsed")
    )
    print("Test run Baseline Loader...")
    # loader.process_file("qasper-dev-v0.3.json")
