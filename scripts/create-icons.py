from PIL import Image, ImageDraw
import os

def create_icon(size, filename):
    """Create a gradient icon with size x size pixels."""
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Corner radius
    radius = size // 8
    
    # Draw gradient background
    for y in range(size):
        for x in range(size):
            # Purple to blue gradient
            ratio = (x + y) / (2 * size)
            r = int(102 + (118 - 102) * ratio)
            g = int(126 + (75 - 126) * ratio)
            b = int(234 + (162 - 234) * ratio)
            
            # Check if inside rounded rect
            if (x >= radius and x < size - radius) or \
               (y >= radius and y < size - radius) or \
               ((x - radius) ** 2 + (y - radius) ** 2 <= radius ** 2) or \
               ((x - (size - radius)) ** 2 + (y - radius) ** 2 <= radius ** 2) or \
               ((x - radius) ** 2 + (y - (size - radius)) ** 2 <= radius ** 2) or \
               ((x - (size - radius)) ** 2 + (y - (size - radius)) ** 2 <= radius ** 2):
                img.putpixel((x, y), (r, g, b, 255))
    
    # Draw white diamond in center
    center = size // 2
    diamond_size = size // 3
    
    # Draw white checkmark or diamond
    draw.polygon([
        (center, center - diamond_size),
        (center + diamond_size, center),
        (center, center + diamond_size),
        (center - diamond_size, center)
    ], fill=(255, 255, 255, 230))
    
    # Draw white circle in center
    circle_radius = max(2, size // 10)
    draw.ellipse([
        (center - circle_radius, center - circle_radius),
        (center + circle_radius, center + circle_radius)
    ], fill=(255, 255, 255, 255))
    
    img.save(filename, 'PNG')
    print(f"Created {filename}")

# Create icons
os.makedirs('extension/icons', exist_ok=True)
create_icon(16, 'extension/icons/icon16.png')
create_icon(48, 'extension/icons/icon48.png')
create_icon(128, 'extension/icons/icon128.png')

print("✓ All icons created successfully!")
