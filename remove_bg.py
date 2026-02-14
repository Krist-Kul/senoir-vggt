import os
import argparse
import io
from rembg import remove
from PIL import Image
from pathlib import Path
from pillow_heif import register_heif_opener

# This allows Pillow to understand .heic files
register_heif_opener()

def process_image(input_path, output_path):
    """Processes a single image: removes background and saves as high-res PNG."""
    try:
        # Load image and keep original metadata/resolution
        with Image.open(input_path) as img:
            # We convert to RGB/RGBA because rembg works best with standard pixel data
            # HEIC is often in YCbCr; converting ensures the AI reads it correctly
            img_converted = img.convert("RGBA")
            
            # Save to a byte buffer to pass to rembg
            buffer = io.BytesIO()
            img_converted.save(buffer, format="PNG")
            input_data = buffer.getvalue()
            
            # Remove background
            output_data = remove(input_data)
            
            # Save the final result
            with open(output_path, 'wb') as o:
                o.write(output_data)
                
        print(f"✓ Processed: {input_path.name} ({img.size[0]}x{img.size[1]})")
    except Exception as e:
        print(f"✗ Failed to process {input_path.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Remove background (Supports JPG, PNG, HEIC)")
    parser.add_argument("-i", "--input", required=True, help="Input file or directory")
    parser.add_argument("-o", "--output", required=True, help="Output file or directory")
    
    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    # Define supported formats
    valid_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.heic', '.heif'}

    if input_path.is_dir():
        if not output_path.exists():
            output_path.mkdir(parents=True)
        
        # Gather all matching files (case-insensitive)
        files_to_process = [
            f for f in input_path.iterdir() 
            if f.suffix.lower() in valid_extensions
        ]

        if not files_to_process:
            print(f"No valid images found in {input_path}")
            return

        print(f"Processing {len(files_to_process)} images...")
        for file in files_to_process:
            # Output is always PNG to support transparency
            out_file = output_path / f"{file.stem}_no_bg.png"
            process_image(file, out_file)

    elif input_path.is_file():
        # Handle single file case
        if output_path.suffix.lower() != '.png':
            # If output is a directory or lacks .png, force it to png
            if not output_path.suffix:
                output_path.mkdir(parents=True, exist_ok=True)
                output_path = output_path / f"{input_path.stem}_no_bg.png"
            else:
                output_path = output_path.with_suffix('.png')
        
        process_image(input_path, output_path)
    
    else:
        print(f"Error: Path {args.input} not found.")

if __name__ == "__main__":
    main()