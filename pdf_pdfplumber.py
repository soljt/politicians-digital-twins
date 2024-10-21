import pdfplumber

def extract_pdf_structure(pdf_path):
    # Open the PDF document
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            print(f"--- Page {page_num + 1} ---\n")  # Print page number

            # Extract the detailed layout of the page
            page_data = page.extract_words(use_text_flow=True, extra_attrs=["size"])

            for item in page_data:
                font_size = item['size']  # Extract font size
                text = item['text']       # Extract text

                if text.strip():
                    # Print the font size and corresponding text
                    print(f"Font Size: {font_size}, Text: {text[:80]}")  # Truncate long text for display
                    print("-" * 80)

if __name__ == "__main__":
    # Path to the PDF file
    pdf_path = "ssrn_papers/ssrn_pdfs/ssrn-4266038.pdf"
    
    # Extract and expose the structure
    extract_pdf_structure(pdf_path)