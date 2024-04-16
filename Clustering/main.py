

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
from K_Means import KMeans



# Load sample image
image = Image.open("sample.jpg")
image_data = np.array(image, dtype=np.float32) / 255.0
height, width, channels = image_data.shape

# Reshape image data into a 2D array
data = image_data.reshape((height * width, channels))


# to test it for differnt k values
def do_main(k):
    # Create and fit KMeans object
    #k = 8
    kmeans = KMeans(k)

    # Measure the execution time
    start_time = time.time()
    kmeans.fit(data)
    end_time = time.time()
    execution_time = end_time - start_time

    # Calculate clustering error
    clustering_error = kmeans.calculate_error(data)

# Assign each pixel to its corresponding cluster centroid
    cluster_labels = kmeans.predict(data)
    clustered_data = kmeans.centroids[cluster_labels]
    clustered_image = clustered_data.reshape((height, width, channels))

# Display original and quantized images
    plt.clf()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image_data)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    axes[1].imshow(clustered_image)
    axes[1].set_title("clustered_image (k = {})".format(k))
    axes[1].axis("off")
    #plt.show()
    plt.imsave(f"figs2/K {k} - KMeans.png",clustered_image)
    return clustering_error, execution_time
    # print("Clustering Error: {:.4f}".format(clustering_error))
    # print("Computational Time: {:.2f} seconds".format(execution_time))

'''
for k in range(2,21):
    clustering_error, execution_time = do_main(k)
    print(f"Error rate at K= {k}: {clustering_error}     Time: {execution_time}")
'''

kmeans = KMeans(2)
kmeans.find_optimum_k(data)
