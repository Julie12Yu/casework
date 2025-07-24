import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from collections import Counter

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_process_data(json_file_path):
    """
    Load JSON data and process it for UMAP visualization
    """
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(data)
    
    # Extract categories (handle nested structure)
    categories = []
    for case in data:
        if isinstance(case['category']['category'], list) and len(case['category']['category']) > 0:
            categories.append(case['category']['category'][0])
        else:
            categories.append('Unknown')
    
    df['main_category'] = categories
    
    return df, data

def create_text_features(summaries):
    """
    Create TF-IDF features from case summaries
    """
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )
    
    # Fit and transform the summaries
    tfidf_features = vectorizer.fit_transform(summaries)
    
    return tfidf_features.toarray(), vectorizer

def perform_umap(features, n_neighbors=15, min_dist=0.1, n_components=2, random_state=42):
    """
    Perform UMAP dimensionality reduction
    """
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Initialize UMAP
    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state,
        metric='cosine'
    )
    
    # Fit and transform
    embedding = reducer.fit_transform(features_scaled)
    
    return embedding, reducer

def create_visualization(embedding, categories, titles):
    """
    Create the UMAP visualization with color coding
    """
    # Count cases per category
    category_counts = Counter(categories)
    
    # Define category colors with dramatic differences
    category_colors = {
        'Consumer Protection': '#1f77b4',
        'Antitrust': '#ff7f0e', 
        'IP Law': '#2ca02c',
        'Privacy and Data Protection': '#d62728',
        'Tort': '#9467bd',
        'Justice and Equity': '#8c564b',
        'Unrelated': '#e377c2',
        'AI in Legal Proceedings': '#7f7f7f'
    }
    
    # Create single plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Plot colored by category
    unique_categories = list(set(categories))
    
    for category in unique_categories:
        mask = np.array(categories) == category
        count = category_counts[category]
        label_with_count = f"{category} ({count})"
        
        ax.scatter(
            embedding[mask, 0], 
            embedding[mask, 1],
            c=category_colors.get(category, '#2C3E50'),
            label=label_with_count,
            alpha=0.7,
            s=60
        )
    
    ax.set_title('Legal Cases UMAP - Colored by Category', fontsize=18, fontweight='bold')
    ax.set_xlabel('UMAP Dimension 1', fontsize=14)
    ax.set_ylabel('UMAP Dimension 2', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # Remove plt.show() to avoid displaying the plot
    
    return fig

def print_cluster_analysis(embedding, categories, titles):
    """
    Print basic analysis of the clusters
    """
    print("=== CLUSTER ANALYSIS ===")
    print(f"Total cases: {len(categories)}")
    print(f"Embedding shape: {embedding.shape}")
    print("\nCategory distribution:")
    
    category_counts = Counter(categories)
    for category, count in category_counts.most_common():
        percentage = (count / len(categories)) * 100
        print(f"  {category}: {count} cases ({percentage:.1f}%)")
    
    print(f"\nUMAP embedding range:")
    print(f"  X: {embedding[:, 0].min():.2f} to {embedding[:, 0].max():.2f}")
    print(f"  Y: {embedding[:, 1].min():.2f} to {embedding[:, 1].max():.2f}")

def main(json_file_path):
    """
    Main function to run the complete UMAP visualization pipeline
    """
    print("Loading and processing data...")
    df, data = load_and_process_data(json_file_path)
    
    # Extract data for processing
    summaries = [case['summary'] for case in data]
    titles = [case['title'] for case in data]
    categories = df['main_category'].tolist()
    
    print(f"Loaded {len(summaries)} cases")
    
    # Create text features
    print("Creating TF-IDF features...")
    features, vectorizer = create_text_features(summaries)
    print(f"Created {features.shape[1]} features")
    
    # Perform UMAP
    print("Performing UMAP dimensionality reduction...")
    embedding, reducer = perform_umap(features)
    
    # Create visualization
    print("Creating visualization...")
    fig = create_visualization(embedding, categories, titles)
    
    # Print analysis
    print_cluster_analysis(embedding, categories, titles)
    
    # Save the plot
    fig.savefig('legal_cases_umap.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'legal_cases_umap.png'")
    
    return embedding, categories, titles

# Example usage
if __name__ == "__main__":
    # Replace 'your_data.json' with the path to your JSON file
    json_file_path = 'legal_cases.json'
    
    try:
        embedding, categories, titles = main(json_file_path)
        print("UMAP visualization completed successfully!")
        
    except FileNotFoundError:
        print(f"Error: Could not find the file '{json_file_path}'")
        print("Please make sure the file exists and the path is correct.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Additional function to explore specific clusters interactively
def explore_cluster(embedding, categories, titles, summaries, x_range, y_range):
    """
    Explore cases within a specific region of the UMAP plot
    """
    mask = ((embedding[:, 0] >= x_range[0]) & (embedding[:, 0] <= x_range[1]) &
            (embedding[:, 1] >= y_range[0]) & (embedding[:, 1] <= y_range[1]))
    
    cluster_cases = np.where(mask)[0]
    
    print(f"\n=== Cases in region X:{x_range}, Y:{y_range} ===")
    print(f"Found {len(cluster_cases)} cases")
    
    for i, case_idx in enumerate(cluster_cases[:10]):  # Show first 10
        print(f"\n{i+1}. {titles[case_idx]}")
        print(f"   Category: {categories[case_idx]}")
        print(f"   Summary: {summaries[case_idx][:200]}...")
        
    if len(cluster_cases) > 10:
        print(f"\n... and {len(cluster_cases) - 10} more cases")

# Required packages (install with pip):
# pip install numpy pandas matplotlib seaborn scikit-learn umap-learn