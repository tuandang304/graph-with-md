import os
import json
from tqdm import tqdm

class QasperLoader:
    """
    Component 1: Reads Qasper JSON data and converts it to clean Markdown format
    with embedded Figure/Table caption descriptions.
    """
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = input_dir
        self.output_dir = output_dir

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    def process_file(self, filename: str):
        """Process one JSON file (e.g. qasper-dev-v0.3.json) into multiple .md files."""
        filepath = os.path.join(self.input_dir, filename)

        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return

        print(f"Loading file {filepath} (this may use some RAM)...")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"Converting {len(data)} papers from {filename}...")
        for paper_id, content in tqdm(data.items()):
            markdown_text = self._build_markdown(title=content.get('title', ''),
                                                 abstract=content.get('abstract', ''),
                                                 full_text=content.get('full_text', []),
                                                 figures=content.get('figures_and_tables', []))

            output_filepath = os.path.join(self.output_dir, f"{paper_id}.md")
            with open(output_filepath, 'w', encoding='utf-8') as out_f:
                out_f.write(markdown_text)

    def _build_markdown(self, title: str, abstract: str, full_text: list, figures: list) -> str:
        md = []

        # 1. Title
        md.append(f"# {title}\n")

        # 2. Abstract
        if abstract:
            md.append(f"## Abstract\n{abstract}\n")

        # 3. Full text sections
        for section in full_text:
            section_name = section.get('section_name', 'Unnamed Section')
            md.append(f"## {section_name}\n")

            paragraphs = section.get('paragraphs', [])
            for p in paragraphs:
                md.append(f"{p}\n")
            md.append("\n")  # Blank line

        # 4. Workaround for Vision LLM (text-only) via captions
        if figures:
            md.append("## Figures and Tables\n")
            for fig in figures:
                file_name = fig.get('file', 'unknown')
                caption = fig.get('caption', 'No description')
                md.append(f"**[Attached figure: {file_name}]**\n> Description: {caption}\n")

        return "\n".join(md)

if __name__ == "__main__":
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    loader = QasperLoader(
        input_dir=os.path.join(_root, "data", "raw"),
        output_dir=os.path.join(_root, "data", "parsed")
    )
    # Test on dev set (small enough for quick format check)
    print("Test run component Loader...")
    loader.process_file("qasper-dev-v0.3.json")
