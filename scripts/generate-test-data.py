#!/usr/bin/env python3
"""
Generate sample test images for Asala testing.
"""
from PIL import Image, ImageDraw, ImageFont
import os

def create_gradient_image(width, height, filename, title=""):
    """Create a gradient test image."""
    img = Image.new('RGB', (width, height))
    pixels = img.load()
    
    for y in range(height):
        for x in range(width):
            # Create a nice gradient
            r = int(255 * (x / width))
            g = int(255 * (y / height))
            b = int(255 * (1 - (x / width)))
            pixels[x, y] = (r, g, b)
    
    # Add text
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()
    
    text = title or f"Test Image {width}x{height}"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    # Draw text with outline
    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        draw.text((x + dx, y + dy), text, font=font, fill='black')
    draw.text((x, y), text, font=font, fill='white')
    
    img.save(filename, quality=95)
    print(f"Created {filename}")

def create_pattern_image(width, height, filename):
    """Create a pattern test image."""
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw grid pattern
    grid_size = 50
    for x in range(0, width, grid_size):
        draw.line([(x, 0), (x, height)], fill='lightgray', width=1)
    for y in range(0, height, grid_size):
        draw.line([(0, y), (width, y)], fill='lightgray', width=1)
    
    # Draw some shapes
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange']
    for i, color in enumerate(colors):
        x = 100 + i * 150
        y = 100 + i * 100
        draw.ellipse([x, y, x+100, y+100], fill=color, outline='black', width=2)
    
    img.save(filename, quality=95)
    print(f"Created {filename}")

def create_transparent_image(width, height, filename):
    """Create PNG with transparency."""
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw checkerboard pattern
    square_size = 40
    for y in range(0, height, square_size):
        for x in range(0, width, square_size):
            if ((x // square_size) + (y // square_size)) % 2 == 0:
                draw.rectangle([x, y, x+square_size, y+square_size], 
                             fill=(100, 150, 200, 200))
    
    # Draw circle in center
    center_x, center_y = width // 2, height // 2
    radius = 100
    draw.ellipse([center_x-radius, center_y-radius, 
                  center_x+radius, center_y+radius], 
                 fill=(255, 100, 100, 230), outline='black', width=3)
    
    img.save(filename)
    print(f"Created {filename}")

def create_signature_image(width, height, filename):
    """Create an image that looks like a document."""
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Add lines to simulate text
    margin = 50
    line_height = 30
    y = margin
    
    while y < height - margin:
        line_width = width - 2 * margin - (y % 100)
        draw.line([(margin, y), (margin + line_width, y)], 
                 fill='gray', width=2)
        y += line_height
    
    # Add a "signature" area
    sig_y = height - 150
    draw.text((margin, sig_y), "Signed:", fill='black')
    draw.line([(margin + 100, sig_y + 20), (width - margin, sig_y + 20)], 
             fill='blue', width=2)
    
    img.save(filename, quality=95)
    print(f"Created {filename}")

# Create directories
os.makedirs('test-data/original', exist_ok=True)
os.makedirs('test-data/signed', exist_ok=True)
os.makedirs('test-data/tampered', exist_ok=True)

print("Generating sample test images...")
print("=" * 50)

# Create various test images
create_gradient_image(1920, 1080, 'test-data/original/sample-landscape.jpg', 
                     "Landscape Test Image")
create_gradient_image(1080, 1920, 'test-data/original/sample-portrait.jpg', 
                     "Portrait Test Image")
create_pattern_image(1024, 1024, 'test-data/original/sample-pattern.jpg')
create_transparent_image(512, 512, 'test-data/original/sample-transparent.png')
create_signature_image(800, 600, 'test-data/original/sample-document.jpg')

print("=" * 50)
print("✓ All sample images created successfully!")
print("\nYou can now:")
print("  1. Sign these images: asala sign test-data/original/sample-landscape.jpg --key keys/private.pem")
print("  2. Verify them: asala verify test-data/original/sample-landscape.jpg")
print("  3. Test tampering detection by modifying images")
