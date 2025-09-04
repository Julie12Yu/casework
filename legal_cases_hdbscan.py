import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import umap
import hdbscan
from collections import Counter
import textwrap
import re

DIR_PATH = 'all_jsons/court_pdfs_text.json'
MIN_CLUSTER_SIZE = 13
MIN_SAMPLES = 4

MIN_DIST = 0.2
N_NEIGHBORS = 13

# WE ARE UMAPPING BEFORE HDBSCAN. UMAP > TSNE RAHHH
IS_UMAP = True

def parse_case_metadata(metadata_string, count):
    """Parse the JSON metadata from the case value string"""
    try:
        # Extract JSON from the markdown code block
        json_match = re.search(r'```json\n(.*?)\n```', metadata_string, re.DOTALL)
        json_str = json_match.group(1)
        metadata = json.loads(json_str)
        return metadata
    except Exception as e:
        print("COUNT: ", count)
        print(f"Warning: Error processing metadata: {e}")
        return {"ai_material": False, "category": ["Unknown"]}

def load_and_process_data():
    with open(DIR_PATH, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Process the new format
    processed_cases = []
    total = 0
    for casename, metadata_string in data.items():
        total += 1
        metadata = parse_case_metadata(metadata_string, total)
        
        # Extract main category
        categories = metadata.get('category', ['Unknown'])
        main_category = categories[0] if isinstance(categories, list) and len(categories) > 0 else 'Unknown'
        
        case_info = {
            'title': casename,
            'summary': casename,  # Using casename as summary since no separate summary is provided
            'main_category': main_category,
            'ai_material': metadata.get('ai_material', False),
            'all_categories': categories
        }
        processed_cases.append(case_info)
    
    df = pd.DataFrame(processed_cases)
    return df, processed_cases

def create_text_features(summaries):
    """Create TF-IDF features from case summaries"""
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )
    
    tfidf_features = vectorizer.fit_transform(summaries)
    return tfidf_features.toarray(), vectorizer

def perform_clustering_and_embedding(features):
    """Perform UMAP + HDBSCAN clustering and return both embedding and cluster labels"""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Step 1: UMAP for dimensionality reduction (optional but often helps HDBSCAN)
    embeddings_for_clustering = features_scaled
    
    if IS_UMAP:
        n_samples = features.shape[0]    
        n_neighbors = min(N_NEIGHBORS, max(2, n_samples - 1))
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=MIN_DIST,
            n_components=10,  # Reduce to 10D for clustering, not 2D
            random_state=42
        )
        embeddings_for_clustering = reducer.fit_transform(features_scaled)
    
    # Step 2: HDBSCAN clustering on the reduced features
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    cluster_labels = clusterer.fit_predict(embeddings_for_clustering)
    
    # Step 3: Create 2D embedding for visualization
    if IS_UMAP:
        viz_reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=MIN_DIST,
            n_components=2,  # 2D for visualization
            random_state=42
        )
        embedding_2d = viz_reducer.fit_transform(features_scaled)
    else:
        # If not using UMAP, use first 2 components of scaled features
        embedding_2d = features_scaled[:, :2]
    
    return embedding_2d, cluster_labels, clusterer

def wrap_text(text, width=60):
    """Helper function to wrap text for better display"""
    return '<br>'.join(textwrap.wrap(str(text), width=width))

def create_interactive_visualization(embedding, cluster_labels, original_categories, titles, summaries, output_file='legal_cases_hdbscan_interactive.html'):
    """Create interactive HDBSCAN visualization showing discovered clusters"""
    
    # Process cluster labels
    unique_clusters = sorted(set(cluster_labels))
    n_clusters = len([c for c in unique_clusters if c != -1])
    n_noise = sum(1 for c in cluster_labels if c == -1)
    
    print(f"HDBSCAN found {n_clusters} clusters and {n_noise} noise points")
    
    # Create cluster names
    cluster_names = []
    for label in cluster_labels:
        if label == -1:
            cluster_names.append("Noise")
        else:
            cluster_names.append(f"Cluster {label}")
    
    # Count cluster sizes
    cluster_counts = Counter(cluster_names)
    
    # Create display labels with counts
    cluster_display = [f"{name} ({cluster_counts[name]})" for name in cluster_names]
    
    # Create color palette
    colors = (px.colors.qualitative.Set3 + 
             px.colors.qualitative.Bold + 
             px.colors.qualitative.Dark24 + 
             px.colors.qualitative.Pastel)
    
    # Map clusters to colors (noise gets gray)
    cluster_color_map = {}
    color_idx = 0
    for cluster in sorted(unique_clusters):
        if cluster == -1:
            cluster_color_map[f"Noise"] = 'lightgray'
        else:
            cluster_color_map[f"Cluster {cluster}"] = colors[color_idx % len(colors)]
            color_idx += 1
    
    # Create DataFrame
    df_plot = pd.DataFrame({
        'x': embedding[:, 0],
        'y': embedding[:, 1],
        'cluster_name': cluster_names,
        'cluster_display': cluster_display,
        'original_category': original_categories,
        'title': titles,
        'summary': summaries,
        'summary_preview': [s[:300] + '...' if len(s) > 300 else s for s in summaries],
        'title_wrapped': [t[:80] + '...' if len(t) > 80 else t for t in titles]
    })
    
    # Wrap summary text
    df_plot['summary_wrapped'] = df_plot['summary_preview'].apply(lambda x: wrap_text(x, width=60))
    
    # Create empty figure
    fig = go.Figure()
    
    # Add scatter trace for each cluster
    for cluster_name in sorted(set(cluster_names)):
        mask = df_plot['cluster_name'] == cluster_name
        cluster_data = df_plot[mask]
        
        if len(cluster_data) > 0:
            count = cluster_counts[cluster_name]
            
            # Special styling for noise points
            if cluster_name == "Noise":
                marker_dict = dict(
                    color='lightgray',
                    size=6,
                    opacity=0.6,
                    line=dict(width=0.5, color='darkgray')
                )
            else:
                marker_dict = dict(
                    color=cluster_color_map[cluster_name],
                    size=8,
                    line=dict(width=0.5, color='white')
                )
            
            fig.add_trace(go.Scatter(
                x=cluster_data['x'],
                y=cluster_data['y'],
                mode='markers',
                marker=marker_dict,
                name=f"{cluster_name} ({count})",
                legendgroup=cluster_name,
                customdata=np.column_stack((
                    cluster_data['cluster_display'],
                    cluster_data['original_category'],
                    cluster_data['title_wrapped'],
                    cluster_data['summary_wrapped']
                )),
                hovertemplate='<b>%{customdata[2]}</b><br><br>' +
                              '<b>Discovered Cluster:</b> %{customdata[0]}<br>' +
                              '<b>Original Category:</b> %{customdata[1]}<br><br>' +
                              '<b>Summary:</b><br>%{customdata[3]}<br>' +
                              '<extra></extra>'
            ))
    
    # Update layout
    title_text = f'HDBSCAN Clustering Results ({n_clusters} clusters, {n_noise} noise points)'
    if IS_UMAP:
        title_text += ' - UMAP + HDBSCAN'
    else:
        title_text += ' - HDBSCAN only'

    fig.update_layout(
        title={
            'text': title_text,
            'x': 0.5,
            'font': {'size': 24}
        },
        xaxis_title='UMAP Component 1' if IS_UMAP else 'Component 1',
        yaxis_title='UMAP Component 2' if IS_UMAP else 'Component 2',
        font=dict(size=12),
        legend=dict(
            title='Discovered Clusters',
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        ),
        width=1200,
        height=800,
        margin=dict(r=250),
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="black",
            font_size=11,
            font_family="Arial",
            align="left"
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    # Save and show
    fig.write_html(output_file)
    print(f"Interactive visualization saved to: {output_file}")
    fig.show()
    
    return fig

def analyze_clusters(cluster_labels, original_categories, titles):
    """Analyze the discovered clusters vs original categories"""
    print("=== CLUSTER ANALYSIS ===")
    
    unique_clusters = sorted(set(cluster_labels))
    n_clusters = len([c for c in unique_clusters if c != -1])
    n_noise = sum(1 for c in cluster_labels if c == -1)
    
    print(f"HDBSCAN Results:")
    print(f"  - Found {n_clusters} clusters")
    print(f"  - {n_noise} points classified as noise")
    print(f"  - Total points: {len(cluster_labels)}")
    
    print(f"\nCluster sizes:")
    cluster_counts = Counter(cluster_labels)
    for cluster_id in sorted(cluster_counts.keys()):
        if cluster_id == -1:
            print(f"  Noise: {cluster_counts[cluster_id]} cases")
        else:
            print(f"  Cluster {cluster_id}: {cluster_counts[cluster_id]} cases")
    
    print(f"\nOriginal category distribution:")
    category_counts = Counter(original_categories)
    for category, count in category_counts.most_common():
        percentage = (count / len(original_categories)) * 100
        print(f"  {category}: {count} cases ({percentage:.1f}%)")
    
    # Analyze cluster composition
    print(f"\nCluster composition by original category:")
    for cluster_id in sorted([c for c in unique_clusters if c != -1]):
        cluster_mask = [i for i, c in enumerate(cluster_labels) if c == cluster_id]
        cluster_categories = [original_categories[i] for i in cluster_mask]
        cluster_cat_counts = Counter(cluster_categories)
        
        print(f"\n  Cluster {cluster_id} ({len(cluster_mask)} cases):")
        for cat, count in cluster_cat_counts.most_common(3):  # Top 3 categories
            pct = (count / len(cluster_mask)) * 100
            print(f"    - {cat}: {count} ({pct:.1f}%)")

def main():
    """Main function to run the complete HDBSCAN clustering pipeline"""
    print("Loading and processing data...")
    df, data = load_and_process_data()
    
    # Extract data for processing
    summaries = [case['summary'] for case in data]
    titles = [case['title'] for case in data]
    categories = df['main_category'].tolist()
    
    print(f"Loaded {len(summaries)} cases")
    
    # Create text features
    print("Creating TF-IDF features...")
    features, vectorizer = create_text_features(summaries)
    print(f"Created {features.shape[1]} features")
    
    # Perform clustering and embedding
    print("Performing UMAP + HDBSCAN clustering...")
    embedding_2d, cluster_labels, clusterer = perform_clustering_and_embedding(features)
    
    # Create interactive visualization
    print("Creating interactive visualization...")
    fig = create_interactive_visualization(embedding_2d, cluster_labels, categories, titles, summaries)
    
    # Analyze results
    analyze_clusters(cluster_labels, categories, titles)
    
    return embedding_2d, cluster_labels, categories, titles, fig

# Example usage
if __name__ == "__main__":    
    try:
        main()
        print("HDBSCAN clustering completed successfully!")
        
    except FileNotFoundError:
        print(f"Error: Could not find the file '{DIR_PATH}'")
        print("Please make sure the file exists and the path is correct.")