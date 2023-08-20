# EigenFace
This project deals with Dimensionality Reduction and Image Classification using PCA and Euclidean Distance.  

# Description
In the realm of data analysis and pattern recognition, the challenge of processing high-dimensional datasets often arises, where conventional visualization techniques become limited due to the human capacity to comprehend information in only three dimensions. The project "Dimensionality Reduction and Image Classification using PCA and Euclidean Distance" addresses this limitation by leveraging Principal Component Analysis (PCA) to reduce the dimensions of images, facilitating their visualization in two or three dimensions. Subsequently, the reduced-dimensional representations are employed for image classification through the application of Euclidean distance-based matching.

Principal Component Analysis (PCA) is a powerful mathematical technique that projects high-dimensional data onto a lower-dimensional subspace, capturing the most significant variations in the data. This project capitalizes on the benefits of PCA to transform high-dimensional image data into a compact and informative representation, while preserving as much variance as possible. The resulting reduced-dimensional representations enable a more intuitive visualization and exploration of the dataset.

The core objective of this project is to facilitate image classification using the reduced-dimensional image representations obtained through PCA. Since humans struggle to perceive and comprehend data beyond three dimensions, the reduction to two or three dimensions becomes crucial for effective data interpretation. The project team employs a dataset containing diverse images, ensuring a wide array of image features and patterns.

The workflow of the project unfolds in several steps:

1. **Data Preparation**: A dataset comprising high-dimensional images is gathered and preprocessed. This involves tasks such as resizing images to a uniform dimension, converting them to grayscale or RGB format, and ensuring consistency across the dataset.

2. **PCA Dimensionality Reduction**: PCA is applied to the preprocessed images, extracting the principal components that capture the most significant variations in the data. These components form a new basis for representing the images in a reduced-dimensional space.

3. **Visualization**: The reduced-dimensional image representations are visualized in two or three dimensions, allowing for an insightful exploration of the dataset. Visual patterns and clusters that might not be apparent in the original high-dimensional space can now be discerned.

4. **Euclidean Distance-based Classification**: To achieve image classification, the project employs Euclidean distance as a metric to calculate the similarity between the reduced-dimensional representations of input images and those in the dataset. The image in the dataset with the closest reduced representation to the input image is identified as the best match, leading to successful image classification.

5. **Evaluation and Results**: The effectiveness of the image classification process is evaluated using appropriate metrics such as accuracy, precision, recall, and F1 score. The project team demonstrates how the combination of PCA-based dimensionality reduction and Euclidean distance-based matching yields robust and accurate image classification results.

In conclusion, the "Dimensionality Reduction and Image Classification using PCA and Euclidean Distance" project showcases the utility of PCA for reducing high-dimensional image data into a more comprehensible space. By employing Euclidean distance-based matching, the project successfully achieves the task of image classification, allowing for efficient categorization and analysis of images. This approach holds significance in fields where visualizing and understanding complex datasets is paramount, such as computer vision, medical imaging, and data-driven decision-making.

# Sample data from Dataset
![Sample images from Dataset](https://github.com/Yeshvendra/EigenFace/blob/main/Output/Sample_Images.png?raw=true)

# Sample data from Eigenfaces
![Sample Eigenfaces](https://github.com/Yeshvendra/EigenFace/blob/main/Output/Eigenfaces.png?raw=true)

# Output when the face was matched
![Output when face matched](https://github.com/Yeshvendra/EigenFace/blob/main/Output/Output_Existing_Image.png?raw=true)

# Output when the face was not matched
![Output when face was not matched](https://github.com/Yeshvendra/EigenFace/blob/main/Output/Output_NonExisting_Image.png?raw=true)
