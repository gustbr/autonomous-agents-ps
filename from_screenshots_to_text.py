import os
import pytesseract
from PIL import Image

def process_images_to_text(folder_path='screenshots', output_file='output.txt'):
    # Get all PNG files in the folder and sort them
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')],
                        key=lambda x: int(x.split('Page ')[1].split('.')[0]))

    # Open output file
    with open(output_file, 'w', encoding='utf-8') as output:
        # Process each image
        for i, image_file in enumerate(image_files, 1):
            try:
                # Print progress
                print(f"Processing image {i}/{len(image_files)}: {image_file}")

                # Open and process image
                image_path = os.path.join(folder_path, image_file)
                image = Image.open(image_path)
                text = pytesseract.image_to_string(image)

                # Write to file with page separator
                output.write(f"\n{'='*50}\nPage {i}\n{'='*50}\n\n")
                output.write(text)
                output.write('\n')

            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")

if __name__ == "__main__":
    process_images_to_text()
    print("\nProcessing complete! Check output.txt for results.")
