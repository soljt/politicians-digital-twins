import fitz  # PyMuPDF

def extract_pdf_structure(pdf_path):
    # Open the PDF document
    doc = fitz.open(pdf_path)
    
    # Iterate over all pages in the PDF
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)  # Load a page
        blocks = page.get_text("dict")["blocks"]  # Extract text blocks in dictionary format
        
        print(f"--- Page {page_num + 1} ---\n")  # Print page number

        # Iterate over blocks of text on the page
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        # Extract text and font size for each span
                        font_size = span["size"]
                        text = span["text"].strip()

                        if text and font_size > 10.5 and font_size < 11 and page_num == 2:  # Filter text based on font size
                            # Print the text along with its font size
                            print(f"Font Size: {font_size}, Text: {text}")  # Truncate long text for display
                            print("-" * 80)

if __name__ == "__main__":
    # Path to the PDF file
    pdf_path = "ssrn_papers/ssrn_pdfs/ssrn-4266038.pdf"
    
    # Extract and expose the structure
    extract_pdf_structure(pdf_path)

