from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
import numpy as np
from base64 import b64encode
from utils import load_image
from segments import segment_image_with_gmm

app = Flask(__name__)

# Adding the b64encode filter to Jinja2
@app.template_filter('b64encode')
def b64encode_filter(data):
    return b64encode(data).decode('utf-8')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded image and number of clusters
        image_file = request.files['image']
        n_clusters = int(request.form['n_clusters'])

        # Load and process the image
        image = load_image(image_file)
        image = image.astype(np.uint8)  # Ensure the image is of type uint8
        image_height, image_width, _ = image.shape
        image_pixels = np.reshape(image, (-1, 3))

        # Perform GMM segmentation
        segmented_map, binary_map = segment_image_with_gmm(image_pixels, image_height, image_width, n_clusters)

        # Convert numpy arrays to images
        segmented_image = Image.fromarray(segmented_map.astype(np.uint8))
        binary_image = Image.fromarray((binary_map * 255).astype(np.uint8))

        # Convert the original image to a PIL Image object
        original_image = Image.fromarray(image)

        # Save images to bytes for display
        segmented_io = io.BytesIO()
        segmented_image.save(segmented_io, 'PNG')
        segmented_io.seek(0)

        binary_io = io.BytesIO()
        binary_image.save(binary_io, 'PNG')
        binary_io.seek(0)

        original_io = io.BytesIO()
        original_image.save(original_io, 'PNG')
        original_io.seek(0)

        # Encode images to base64
        original_image_base64 = b64encode(original_io.getvalue()).decode('utf-8')
        segmented_image_base64 = b64encode(segmented_io.getvalue()).decode('utf-8')
        binary_image_base64 = b64encode(binary_io.getvalue()).decode('utf-8')

        return jsonify({
            'original_image': original_image_base64,
            'segmented_image': segmented_image_base64,
            'binary_image': binary_image_base64
        })

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
