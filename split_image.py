from PIL import Image
import os

def split_image(image_path, grid_size=2):
    """Splits an image into equal tiles based on the grid size."""
    image = Image.open(image_path)
    width, height = image.size
    tile_width = width // grid_size
    tile_height = height // grid_size

    tiles = []
    for i in range(grid_size):
        for j in range(grid_size):
            left = j * tile_width
            upper = i * tile_height
            right = left + tile_width
            lower = upper + tile_height
            tile = image.crop((left, upper, right, lower))
            tiles.append(tile)

    return tiles

# Make sure 'puzzle_image.jpg' is available in your Codespaces
image_path = "puzzle.png"

# Split image
tiles = split_image(image_path)

# Create folder for tiles
tile_folder = "puzzle_tiles"
os.makedirs(tile_folder, exist_ok=True)

# Save tiles and store paths
tile_paths = []
for idx, tile in enumerate(tiles):
    tile_path = os.path.join(tile_folder, f"tile_{idx}.png")
    tile.save(tile_path)
    tile_paths.append(tile_path)

print("Image split into tiles:", tile_paths)
