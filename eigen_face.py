import cv2
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA

faces = {}
with zipfile.ZipFile("attface.zip") as facezip:
    for filename in facezip.namelist():
        if not filename.endswith(".pgm"):
            continue
        with facezip.open(filename) as image:
            faces[filename] = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

fig, axes = plt.subplots(4, 4, sharex = True, sharey = True, figsize = (8, 10))

# Randomly choose 16 images
faceimages = random.choices(list(faces.values()), k = 16)

# Display those 16 images
for i in range(16):
    axes[i%4][i//4].imshow(faceimages[i], cmap="gray")
plt.show()

# Get the pixel size of each image
faceshape = list(faces.values())[0].shape
print("Face image shape (pixel size of each image):", faceshape)

# List the first 5 file names which were read
print(list(faces.keys())[:5])

# Create classes of each image
classes = set(filename.split("/")[0] for filename in faces.keys())
print("Number of classes:", len(classes))
print("Number of pictures:", len(faces))

# Consider classes 1-39 for eigenfaces and keep class 40 and
# image 10 of class 39 as test element.
facematrix = []
facelabel = []
for key,val in faces.items():
    if key.startswith("s40/"):
        continue
    if key == "s39/10.pgm":
        continue
    facematrix.append(val.flatten())
    facelabel.append(key.split("/")[0])

# Convert to matrix
facematrix = np.array(facematrix)

# Apply PCA
pca = PCA().fit(facematrix)

print(pca.explained_variance_ratio_)

# Take only first k principal components as eigenfaces
n_components = 50
eigenfaces = pca.components_[:n_components]

# Show the first 16 eigenfaces
fig, axes = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(8,10))
for i in range(16):
    axes[i%4][i//4].imshow(eigenfaces[i].reshape(faceshape), cmap="gray")
plt.show()

# Generate weights as a KxN matrix where K is the number of eigenfaces and 
# N the number of samples
weights = eigenfaces @ (facematrix - pca.mean_).T

# Test on test image of existing class
query_exist = faces["s39/10.pgm"].reshape(1, -1)
query_weight_exist = eigenfaces @ (query_exist - pca.mean_).T
euclidean_distance_exist = np.linalg.norm(weights - query_weight_exist, axis=0)
best_match_exist = np.argmin(euclidean_distance_exist)

# Visualize Result
fig, axes = plt.subplots(1,2,sharex=True, sharey=True, figsize=(8,6))
axes[0].imshow(query_exist.reshape(faceshape), cmap="gray")
axes[0].set_title("Query")
axes[1].imshow(facematrix[best_match_exist].reshape(faceshape), cmap="gray")
axes[1].set_title("Best match")
plt.show()

# Test on test image of non-existing class
query_nonexist = faces["s40/10.pgm"].reshape(1, -1)
query_weight_nonexist = eigenfaces @ (query_nonexist - pca.mean_).T
euclidean_distance_nonexist = np.linalg.norm(weights - query_weight_nonexist, axis=0)
best_match_nonexist = np.argmin(euclidean_distance_nonexist)

# Visualize Result
fig, axes = plt.subplots(1,2,sharex=True, sharey=True, figsize=(8,6))
axes[0].imshow(query_nonexist.reshape(faceshape), cmap="gray")
axes[0].set_title("Query")
axes[1].imshow(facematrix[best_match_nonexist].reshape(faceshape), cmap="gray")
axes[1].set_title("Best match")
plt.show()
