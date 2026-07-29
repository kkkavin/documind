from langchain_community.document_loaders import PyPDFLoader

def load_pdf(file):
    loader = PyPDFLoader(f"data/{file}.pdf")
    loaded_list = loader.load()
    return loaded_list

if __name__ == "__main__":
    raw_text = load_pdf('report')
    if raw_text:
        print("\n--- RAW TEXT PREVIEW ---")
        print("Metadata:", raw_text[0].metadata)
        print("Text Preview:\n", raw_text[0].page_content)

