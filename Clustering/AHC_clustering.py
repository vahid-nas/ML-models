

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time


class Cluster:
    def __init__(self, pixels,pixels_indices):
        self.pixels = pixels
        self.pixels_indices = pixels_indices
        self.center = np.mean(self.pixels, axis=0)

    def merge(self, cluster):
        self.pixels += cluster.pixels
        self.pixels_indices += cluster.pixels_indices
        self.center = np.mean(self.pixels, axis=0)

class AgglomerativeClustering:
    def __init__(self, image_path, num_clusters):
        self.image_path = image_path
        self.num_clusters = num_clusters
        # self.execution_time = 0
        self.pixels = None
        self.pixels_indices = None
        self.clusters = []
        self.error_list = []
        self.K_Values = []
        self.time_list = []

    def load_image(self):
        img = Image.open(self.image_path)
        self.pixels = np.array(img, dtype=np.float64) / 255
        self.flattened_image = self.pixels.reshape(-1, self.pixels.shape[-1])

    def initialize_clusters(self):

        #flattened_image = self.pixels.reshape(-1, self.pixels.shape[-1])
        pixels_scaled = [[int(value * 255) for value in pixel] for pixel in self.flattened_image]

        height, width, _ = self.pixels.shape
        print(f'Height: {height}, Width: {width}, Total: {height*width}')
        # for i in range (0,height*width):
        #     row = i // width
        #     col = i % width

        rgb_groups = [[],[],[]]
        for i_rgb in range(0,3):
            rgb_list = [sublist[i_rgb] for sublist in pixels_scaled]
            for i_sub_group in range(16):
                minimum = i_sub_group * 16
                maximum = ((i_sub_group+1) * 16) - 1
                values_within_range = [pixel_index for pixel_index,value in enumerate(rgb_list) if minimum <= value <= maximum]
                rgb_groups[i_rgb].append(values_within_range)

        a = 1
        n_init_clusters = 0
        init_clusters = []
        for r_list in rgb_groups[0]:
            for g_list in rgb_groups[1]:
                for b_list in rgb_groups[2]:
                    intersection = set(r_list).intersection(g_list).intersection(b_list)
                    intersection_list = list(intersection)
                    init_clusters.append(intersection_list)

        init_clusters = [sublist for sublist in init_clusters if sublist]

        for pixel_groups_indices in init_clusters:
            pixels = [self.flattened_image[index] for index in pixel_groups_indices]
            cluster = Cluster(pixels,pixel_groups_indices)
            self.clusters.append(cluster)

        a = 1
        print(len(self.clusters))



    def compute_distance(self, cluster1, cluster2):
        return np.linalg.norm(cluster1.center - cluster2.center)

    def merge_closest_clusters(self):
        min_distance = float('inf')
        merge_indices = (0, 1)

        for i in range(len(self.clusters)):
            for j in range(i + 1, len(self.clusters)):
                distance = self.compute_distance(self.clusters[i], self.clusters[j])
                if distance < min_distance:
                    min_distance = distance
                    merge_indices = (i, j)

        i, j = merge_indices
        self.clusters[i].merge(self.clusters[j])
        del self.clusters[j]

    def run(self):
        self.load_image()
        self.initialize_clusters()


        start_time = time.time()
        while len(self.clusters) > self.num_clusters:
            self.merge_closest_clusters()
            if len(self.clusters) < 51:
                self.time_list.append(time.time()-start_time)
                self.error_list.append(self.calculate_error())
                self.K_Values.append(len(self.clusters))
            if len(self.clusters) < 13:
                self.visualize()
        # end_time = time.time()
        # self.execution_time = end_time - start_time

    def calculate_error(self):
        #possible_centroids = self.centroids[self.cluster_labels]
        #flattened_image = self.pixels.reshape(-1, self.pixels.shape[-1])
        clustered_image = np.zeros_like(self.flattened_image)
        for cluster in self.clusters:
            clustered_image[cluster.pixels_indices] = cluster.center
        error = np.mean(np.square(self.flattened_image - clustered_image))
        return error


    def visualize_clusters(self):
        height, width, _ = self.pixels.shape
        clustered_pixels = np.zeros_like(self.pixels)

        for cluster in self.clusters:
            for pixel in cluster.pixels:
                row, col = np.where(np.all(self.pixels == pixel, axis=2))
                clustered_pixels[row, col] = cluster.center

        plt.imshow(clustered_pixels)
        plt.axis('off')
        plt.show()

    def visualize(self):
        height, width, _ = self.pixels.shape
        clustered_pixels = np.zeros_like(self.pixels)

        for cluster in self.clusters:
            for index in cluster.pixels_indices:
                row = index // width
                col = index % width
                clustered_pixels[row][col] = cluster.center
        plt.imshow(clustered_pixels)
        plt.imsave(f"figs/K {len(self.clusters)} - agglo.png", clustered_pixels)
        plt.axis('off')
        #plt.show()


# Usage example
image_path = 'sample.jpg'
num_clusters = 2

clustering = AgglomerativeClustering(image_path, num_clusters)
clustering.run()

# To run print the values of errors and time
for i in range(len(clustering.K_Values)):
    print(f"Error rate at K= {clustering.K_Values[i]}: {clustering.error_list[i]}     Time: {clustering.time_list[i]}")

plt.clf()
plt.plot(clustering.K_Values, clustering.error_list, 'bx-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Clustering Error')
plt.title('Elbow Method')
plt.show()
#clustering.visualize_clusters()
#clustering.visualize()


