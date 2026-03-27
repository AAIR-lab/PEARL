import cv2
import numpy as np

def normalize_coords(coords, img_shape):
    height, width = img_shape[:2]
    return [(x / width, y / height) for x, y in coords]

def detect_shapes_and_generate_txt(image_path, output_txt_path):
    # Read the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    with open(output_txt_path, 'w') as file:
        # Detect balls (assuming they're small circles)
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            area = cv2.contourArea(contour)
            if len(approx) > 8 and area < 500:  # Heuristic for circles
                (x, y), radius = cv2.minEnclosingCircle(contour)
                norm_radius = radius / max(img.shape[:2])
                file.write(f"ball {norm_radius:.2f}\n\n")

        # Define borders
        file.write("# origin is top left, x increases horizontally and y increases vertically\n\n")
        file.write("# top\n")
        file.write("polygon 0.0 0.0 0.0 0.01 1.0 0.01 1.0 0.0\n")
        file.write("# left\n")
        file.write("polygon 0.0 0.0 0.01 0.0 0.01 1.0 0.0 1.0\n")
        file.write("# bottom\n")
        file.write("polygon 0.0 1.0 0.0 0.99 1.0 0.99 1.0 1.0\n")
        file.write("# right\n")
        file.write("polygon 1.0 1.0 0.99 1.0 0.99 0.0 1.0 0.0\n\n")
        
        # Detect polygons
        file.write("# detected obstacles\n")
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            if 4 <= len(approx) <= 8:  # Detect polygons with 4 to 8 sides
                coords = [tuple(pt[0]) for pt in approx]
                norm_coords = normalize_coords(coords, img.shape)
                coord_str = ' '.join(f"{x:.2f} {y:.2f}" for x, y in norm_coords)
                file.write(f"polygon {coord_str}\n")

    print(f"TXT file generated at {output_txt_path}")

# Example usage
detect_shapes_and_generate_txt('Screenshot 2025-02-25 at 10.08.23 AM.png', 'output.txt')
