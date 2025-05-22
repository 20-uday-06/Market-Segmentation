import os
import re
import json
import fitz  # PyMuPDF
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("extraction.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path):
    """Extract all text content from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        return ""

def extract_startup_idea(text):
    """Extract the startup idea using multiple pattern matching approaches."""
    # Common patterns for startup idea identification
    patterns = [
        r"Startup idea name is:\s*\*{2,}(.*?)\*{2,}",  # Format with asterisks
        r"\\title\{\s*(.*?)\s*\}",  # LaTeX title format
        r"(?i)startup idea[^\n]*?[:]\s*(.*?)(?=\n\n|\Z)",  # Generic format
    ]
    
    for pattern in patterns:
        matches = re.search(pattern, text, re.DOTALL)
        if matches:
            idea = matches.group(1).strip()
            # Clean up formatting
            idea = re.sub(r'\*+', '', idea)  # Remove asterisks
            idea = re.sub(r'\s+', ' ', idea)  # Normalize whitespace
            return idea.strip()
    
    # Backup approach: look at first few lines for title-like content
    first_lines = text.split('\n')[:10]
    for line in first_lines:
        line = line.strip()
        if len(line) > 20 and "table" not in line.lower() and "market" not in line.lower():
            return line
    
    return None

def extract_market_segmentation(text):
    """Extract the 16-point market segmentation table."""
    # First locate the market segmentation section
    section_patterns = [
        r"Market Segmentation.*?(?=\n\n\w+|\Z)",
        r"Market Segmentation Table.*?(?=\n\n\w+|\Z)",
    ]
    
    market_section = None
    for pattern in section_patterns:
        matches = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            market_section = matches.group(0)
            break
    
    if not market_section:
        market_section = text  # Use full text if section not found
    
    # Extract points using multiple methods
    points_dict = {}
    
    # Method 1: Numbered list format (1., 2., etc.)
    number_pattern = r'(\d+)[\.\)]\s*(.*?)(?=\s*\d+[\.\)]|\n\n|\Z)'
    points = re.findall(number_pattern, market_section, re.DOTALL)
    
    if points:
        for num, content in points:
            try:
                num = int(num.strip())
                if 1 <= num <= 16:
                    content = content.strip()
                    content = re.sub(r'\s+', ' ', content)
                    points_dict[num] = content
            except ValueError:
                continue
    
    # Method 2: Table format with pipe separators
    if len(points_dict) < 16:
        table_pattern = r"\|\s*(\d+)\s*\|\s*(.*?)\s*\|"
        table_matches = re.findall(table_pattern, market_section)
        
        for num, content in table_matches:
            try:
                num = int(num.strip())
                if 1 <= num <= 16 and (num not in points_dict or not points_dict[num]):
                    content = content.strip()
                    content = re.sub(r'\s+', ' ', content)
                    points_dict[num] = content
            except ValueError:
                continue
    
    # Method 3: LaTeX table format
    if len(points_dict) < 16:
        latex_pattern = r"\\hline\s*(\d+)\s*&\s*(.*?)\s*\\\\"
        latex_matches = re.findall(latex_pattern, market_section)
        
        for num, content in latex_matches:
            try:
                num = int(num.strip())
                if 1 <= num <= 16 and (num not in points_dict or not points_dict[num]):
                    content = content.strip()
                    content = re.sub(r'\s+', ' ', content)
                    points_dict[num] = content
            except ValueError:
                continue
    
    # Ensure we have exactly 16 points
    result = []
    for i in range(1, 17):
        if i in points_dict and points_dict[i]:
            result.append(points_dict[i])
        else:
            result.append(f"Missing point {i}")
    
    # Validate: require at least 8 real points (50%)
    real_points = [p for p in result if not p.startswith("Missing point")]
    return result if len(real_points) >= 8 else None

def process_pdfs(pdf_dir):
    """Process all PDFs in the directory to extract data."""
    pdf_dir = Path(pdf_dir)
    output_data = []
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_dir}")
        return output_data, 0
    
    total_pdfs = len(pdf_files)
    
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            logger.info(f"Processing {pdf_path.name}")
            
            text = extract_text_from_pdf(pdf_path)
            if not text:
                logger.warning(f"No text extracted from {pdf_path.name}")
                continue
            
            startup_idea = extract_startup_idea(text)
            if not startup_idea:
                logger.warning(f"Failed to extract startup idea from {pdf_path.name}")
            
            market_segmentation = extract_market_segmentation(text)
            if not market_segmentation:
                logger.warning(f"Failed to extract market segmentation from {pdf_path.name}")
            
            if startup_idea and market_segmentation:
                output_data.append({
                    "pdf_file": pdf_path.name,
                    "startup_idea": startup_idea,
                    "market_segmentation": market_segmentation
                })
                logger.info(f"Successfully extracted data from {pdf_path.name}")
            else:
                logger.warning(f"Incomplete data extracted from {pdf_path.name}")
        
        except Exception as e:
            logger.error(f"Error processing {pdf_path.name}: {str(e)}")
    
    return output_data, total_pdfs

def save_output(data, output_dir):
    """Save extracted data in multiple formats."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if not data:
        logger.warning("No data to save")
        return None
    
    # Save raw JSON data
    with open(output_dir / "market_segmentation_data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Save CSV with flattened structure for easier viewing
    flat_data = []
    for item in data:
        row = {
            "pdf_file": item["pdf_file"],
            "startup_idea": item["startup_idea"],
        }
        for i, point in enumerate(item["market_segmentation"], 1):
            row[f"segment_point_{i}"] = point
        flat_data.append(row)
    
    df = pd.DataFrame(flat_data)
    df.to_csv(output_dir / "market_segmentation_data.csv", index=False)
    
    # Save formatted training data for Gemma fine-tuning
    training_data = create_training_data(data)
    with open(output_dir / "gemma_training_data.json", "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved output files to {output_dir}")
    return df

def create_training_data(data):
    """Format the data as training examples for Gemma fine-tuning."""
    training_examples = []
    
    for item in data:
        startup_idea = item["startup_idea"]
        segmentation_points = item["market_segmentation"]
        
        # Format as a table
        table_output = "Market Segmentation Table:\n\n"
        for i, point in enumerate(segmentation_points, 1):
            if not point.startswith("Missing point"):
                table_output += f"{i}. {point}\n"
            else:
                table_output += f"{i}. [Point {i}]\n"  # Placeholder
        
        # Create training example in conversation format
        training_example = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a marketing strategist specialized in creating market segmentation analyses for startup ideas. When given a startup idea, you always respond with a table containing exactly 16 market segmentation points, numbered from 1 to 16."
                },
                {
                    "role": "user",
                    "content": f"Create a market segmentation table with 16 points for this startup idea: {startup_idea}"
                },
                {
                    "role": "assistant",
                    "content": table_output
                }
            ]
        }
        training_examples.append(training_example)
    
    return training_examples

def display_sample_table(df):
    """Display a sample of the extracted data in tabular form."""
    if df is None or df.empty:
        return
    
    print("\nSample Market Segmentation Table (first entry):")
    print("-" * 60)
    
    sample_row = df.iloc[0]
    print(f"Startup Idea: {sample_row['startup_idea']}")
    print("\nMarket Segmentation Points:")
    
    for i in range(1, 17):
        col_name = f"segment_point_{i}"
        if col_name in sample_row:
            point = sample_row[col_name]
            if not point.startswith("Missing point"):
                print(f"{i}. {point}")
            else:
                print(f"{i}. [Not extracted]")

def main():
    # Define directories
    pdf_dir = "pdf"
    output_dir = "output"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing PDFs from {pdf_dir}...")
    data, total_pdfs = process_pdfs(pdf_dir)
    
    if data:
        df = save_output(data, output_dir)
        print(f"\nExtraction complete!")
        print(f"Total PDFs processed: {total_pdfs}")
        print(f"PDFs with successfully extracted data: {len(data)}")
        print(f"Success rate: {len(data)/total_pdfs*100:.1f}%")
        
        # Display sample table
        display_sample_table(df)
        
        print("\nOutput files generated:")
        print(f"1. {output_dir}/market_segmentation_data.json - Raw extracted data")
        print(f"2. {output_dir}/market_segmentation_data.csv - CSV format for easy viewing")
        print(f"3. {output_dir}/gemma_training_data.json - Formatted for Gemma fine-tuning")
    else:
        print("No data was extracted from PDFs. Check extraction.log for details.")

if __name__ == "__main__":
    main()
