import sys

def read_cypher_file(file_path, sample_lines=20):
    """
    Reads a .cypher file line by line and prints the first `sample_lines` lines.
    This method is memory-efficient and suitable for large files.
    
    Args:
        file_path (str): Path to the .cypher file.
        sample_lines (int): Number of lines to print as a sample.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                print(line.rstrip())
                if i + 1 >= sample_lines:
                    break
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except UnicodeDecodeError:
        print(f"Encoding error. Try changing the encoding parameter.")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    read_cypher_file('data/KGItalianLegislation.cypher', 11)

if __name__ == "__main__":
    main()
