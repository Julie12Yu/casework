# write a python script that
# 1. load all the pdfs in "data"
# 2. do a simple TF-IDF analysis on the text
# 3. do a tsne plot 
# 4. do a kmeans clustering

import os
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from collections import defaultdict
import json
import matplotlib.pyplot as plt

# Load all the pdfs in "data"
pdf_files = [f for f in os.listdir("data") if f.endswith(".pdf")]
pdf_files = pdf_files[:31]

# Load the pdfs and extract text
# texts = defaultdict(list)
# for pdf_file in pdf_files:
#     with open(f"data/{pdf_file}", "rb") as file:
#         pdf = PyPDF2.PdfReader(file)
#         # Combine all pages into one text
#         full_text = ""
#         for page in pdf.pages:
#             full_text += page.extract_text() + "\n"
#         texts[pdf_file] = full_text

texts = json.load(open("texts.json"))

# # Save the texts for later use
# with open("texts.json", "w") as file:
#     json.dump(texts, file)

# Convert texts to list of documents
documents = list(texts.values())
doc_names = list(texts.keys())

# Do a simple TF-IDF analysis on the text
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Do a tsne plot
tsne = TSNE(n_components=2, random_state=42, init="random")
tsne_result = tsne.fit_transform(tfidf_matrix)

# Print the shape of the results
print(f"Number of documents: {len(documents)}")
print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
print(f"t-SNE result shape: {tsne_result.shape}")

# Print document names
print("\nDocument names:")
for name in doc_names:
    print(f"- {name}")

# Do a kmeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(tfidf_matrix)

# Print the kmeans clustering results
print("\nK-means clustering results:")
for i, label in enumerate(kmeans_labels):
    print(f"Document {doc_names[i]}: Cluster {label}")

# Do a tsne plot with kmeans clusters
tsne_kmeans = TSNE(n_components=2, random_state=42, init="random")
tsne_kmeans_result = tsne_kmeans.fit_transform(tfidf_matrix)

# Print the shape of the kmeans tsne results
print(f"K-means t-SNE result shape: {tsne_kmeans_result.shape}")

# Plot the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=kmeans_labels, cmap='viridis')
plt.title('t-SNE with K-means Clusters')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')

plt.subplot(1, 2, 2)
plt.scatter(tsne_kmeans_result[:, 0], tsne_kmeans_result[:, 1], c=kmeans_labels, cmap='viridis')
plt.title('t-SNE with K-means Clusters')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')

plt.show()

# save the tsne plot
plt.savefig("tsne_plot.png")

