import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from collections import Counter

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

def wrap_text(text, width=50):
    """
    Wrap text to specified width for better display in hover boxes
    """
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 <= width:
            current_line.append(word)
            current_length += len(word) + 1
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word)
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return '<br>'.join(lines)

def create_interactive_visualization(embedding, categories, titles, summaries, output_file='legal_cases_umap_interactive.html'):
    """
    Create an interactive UMAP visualization with hover information
    """
    # Count cases per category
    category_counts = Counter(categories)
    
    # Create DataFrame for Plotly
    df_plot = pd.DataFrame({
        'x': embedding[:, 0],
        'y': embedding[:, 1],
        'category': categories,
        'title': titles,
        'summary': summaries,
        'summary_preview': [s[:300] + '...' if len(s) > 300 else s for s in summaries],
        'title_wrapped': [title[:80] + '...' if len(title) > 80 else title for title in titles]
    })
    
    # Apply text wrapping to summaries
    df_plot['summary_wrapped'] = df_plot['summary_preview'].apply(lambda x: wrap_text(x, width=60))
    
    # Add count to category names
    df_plot['category_with_count'] = df_plot['category'].apply(
        lambda x: f"{x} ({category_counts[x]})"
    )
    
    # Define vibrant colors for categories
    color_map = {
        'Consumer Protection': '#1f77b4',
        'Antitrust': '#ff7f0e', 
        'IP Law': '#2ca02c',
        'Privacy and Data Protection': '#d62728',
        'Tort': '#9467bd',
        'Justice and Equity': '#8c564b',
        'Unrelated': '#e377c2',
        'AI in Legal Proceedings': '#7f7f7f'
    }
    
    # Create the interactive scatter plot
    fig = px.scatter(
        df_plot,
        x='x',
        y='y',
        color='category',
        color_discrete_map=color_map,
        hover_data={
            'x': False,
            'y': False,
            'category': True,
            'title': True,
            'summary_preview': True
        },
        title='Interactive Legal Cases UMAP Visualization',
        labels={
            'x': 'UMAP Dimension 1',
            'y': 'UMAP Dimension 2',
            'category': 'Category'
        }
    )
    
    # Customize hover template with better formatting
    fig.update_traces(
        hovertemplate='<b>%{customdata[1]}</b><br><br>' +
                      '<b>Category:</b> %{customdata[0]}<br><br>' +
                      '<b>Summary:</b><br>%{customdata[2]}<br>' +
                      '<extra></extra>',
        customdata=np.column_stack((df_plot['category'], df_plot['title_wrapped'], df_plot['summary_wrapped'])),
        marker=dict(size=8)  # Make points slightly larger for better visibility
    )
    
    # Update layout for better appearance
    fig.update_layout(
        title={
            'text': 'Interactive Legal Cases UMAP Visualization',
            'x': 0.5,
            'font': {'size': 24}
        },
        xaxis_title='UMAP Dimension 1',
        yaxis_title='UMAP Dimension 2',
        font=dict(size=12),  # Slightly smaller font
        legend=dict(
            title='Category (Count)',
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),
        width=1200,
        height=800,
        margin=dict(r=200),  # Make room for legend
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="black",
            font_size=10,
            font_family="Arial",
            align="left",
            namelength=-1  # Show full text without truncation
        )
    )
    
    # Update legend labels to include counts
    for i, trace in enumerate(fig.data):
        category = trace.name
        count = category_counts[category]
        trace.name = f"{category} ({count})"
    
    # Save as HTML
    fig.write_html(output_file)
    print(f"Interactive visualization saved as '{output_file}'")
    
    # Show the plot
    fig.show()
    
    return fig

def create_github_pages_files(embedding, categories, titles, summaries):
    """
    Create files needed for GitHub Pages deployment
    """
    # Create the interactive visualization
    fig = create_interactive_visualization(embedding, categories, titles, summaries, 'umap_index.html')
    
    # Create a simple README.md for GitHub Pages
    readme_content = """# Legal Cases UMAP Visualization

This is an interactive visualization of legal cases using UMAP (Uniform Manifold Approximation and Projection) dimensionality reduction.

## How to Use
- Hover over any point to see case details including title, category, and summary preview
- Use the legend to filter categories on/off
- Zoom and pan to explore different regions of the plot
- Similar cases cluster together based on text similarity

## About
The visualization uses TF-IDF features extracted from case summaries and reduces them to 2D using UMAP for exploration and pattern discovery.

## View the Visualization
[Click here to view the interactive visualization](./index.html)
"""
    
    with open('README.md', 'w') as f:
        f.write(readme_content)
    
    print("Created README.md for GitHub Pages")
    
    # Create a simple _config.yml for GitHub Pages
    config_content = """title: Legal Cases UMAP Visualization
description: Interactive visualization of legal cases using UMAP dimensionality reduction
theme: jekyll-theme-minimal
"""
    
    with open('_umap_config.yml', 'w') as f:
        f.write(config_content)
    
    print("Created _config.yml for GitHub Pages")

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

def main(json_file_path, create_github_files=True):
    """
    Main function to run the complete interactive UMAP visualization pipeline
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
    
    # Create interactive visualization
    print("Creating interactive visualization...")
    fig = create_interactive_visualization(embedding, categories, titles, summaries)
    
    # Print analysis
    print_cluster_analysis(embedding, categories, titles)
    
    # Create GitHub Pages files if requested
    if create_github_files:
        print("\nCreating GitHub Pages files...")
        create_github_pages_files(embedding, categories, titles, summaries)
        print("\nTo deploy to GitHub Pages:")
        print("1. Create a new repository on GitHub")
        print("2. Upload index.html, README.md, and _config.yml")
        print("3. Enable GitHub Pages in repository settings")
        print("4. Your visualization will be available at: https://yourusername.github.io/yourreponame/")
    
    return embedding, categories, titles, fig

# Example usage
if __name__ == "__main__":
    # Replace 'legal_cases.json' with the path to your JSON file
    json_file_path = 'legal_cases.json'
    
    try:
        embedding, categories, titles, fig = main(json_file_path)
        print("Interactive UMAP visualization completed successfully!")
        
    except FileNotFoundError:
        print(f"Error: Could not find the file '{json_file_path}'")
        print("Please make sure the file exists and the path is correct.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Required packages (install with pip):
# pip install numpy pandas plotly scikit-learn umap-learn