import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from collections import Counter
import textwrap
import re
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

DIR_PATH = 'all_jsons/categorized_cases.json'
MIN_DIST = 0.2
N_NEIGHBORS = 10

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
    """Load JSON data and process it for UMAP visualization"""
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
    """Create TF-IDF features from case summaries (titles in this case)"""
    # Clean and prepare text for TF-IDF
    cleaned_summaries = []    
    vectorizer = TfidfVectorizer(
        max_features=500,  # Reduced since we're working with case names
        stop_words=None,
        ngram_range=(1, 3),  # Include more n-grams for case names
        min_df=1,  # Allow single occurrences
        max_df=0.9
    )
    
    try:
        tfidf_features = vectorizer.fit_transform(cleaned_summaries)
        return tfidf_features.toarray(), vectorizer
    except ValueError as e:
        print(f"Error creating TF-IDF features: {e}")
        # Fallback: create simple character-based features
        from sklearn.feature_extraction.text import CountVectorizer
        char_vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 4), max_features=200)
        char_features = char_vectorizer.fit_transform(summaries)
        return char_features.toarray(), char_vectorizer

def perform_umap(features):
    """Perform UMAP dimensionality reduction"""
    # Adjust parameters based on dataset size
    n_samples = features.shape[0]
    n_neighbors = min(N_NEIGHBORS, max(2, n_samples - 1))
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=MIN_DIST,
    )
    
    embedding = reducer.fit_transform(features_scaled)
    return embedding, reducer

def non_category_silhouette(original_features, new_embedding, categories=None):
    """Evaluate quality using silhouette scores with K-means clustering (no ground truth needed)"""
    print("\n=== SILHOUETTE SCORE ANALYSIS ===")
    
    n_samples = len(original_features)
    max_k = min(10, n_samples // 3)
    
    print(f"Analyzing {n_samples} cases")
    
    try:
        # Test K-means clustering with different k values
        print(f"\n--- K-Means Clustering Analysis ---")
        k_range = range(2, max_k + 1)
        
        for n_clusters in k_range:
            if n_clusters >= n_samples:
                break
                
            # K-means on original space
            kmeans_orig = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            orig_cluster_labels = kmeans_orig.fit_predict(original_features)
            orig_kmeans_sil = silhouette_score(original_features, orig_cluster_labels)
            
            # K-means on new embedding space
            kmeans_new = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            new_cluster_labels = kmeans_new.fit_predict(new_embedding)
            new_kmeans_sil = silhouette_score(new_embedding, new_cluster_labels)
            
            improvement = new_kmeans_sil - orig_kmeans_sil
            print(f"K-means (k={n_clusters}) - Original: {orig_kmeans_sil:.4f}, New: {new_kmeans_sil:.4f}, Δ: {improvement:+.4f}")
        
        # Interpretation
        print(f"\nSilhouette Score Interpretation:")
        print(f"  > 0.7: Strong separation")
        print(f"  0.5-0.7: Reasonable separation") 
        print(f"  0.25-0.5: Weak separation")
        print(f"  < 0.25: Poor separation")
        
    except Exception as e:
        print(f"Error calculating silhouette scores: {e}")

def wrap_text(text, width=60):
    """Helper function to wrap text for better display"""
    return '<br>'.join(textwrap.wrap(str(text), width=width))

def create_interactive_visualization(embedding, categories, titles, ai_materials, all_categories, output_file='legal_full_text_umap.png'):
    """Create interactive UMAP visualization with correct color mapping and save as PNG."""

    # Clean categories
    categories_cleaned = [c.strip() if isinstance(c, str) else 'Unknown' for c in categories]

    # Get unique base categories and assign colors
    unique_base_categories = sorted(set(categories_cleaned))

    # Create color palette
    colors = (px.colors.qualitative.Set3 +
             px.colors.qualitative.Bold +
             px.colors.qualitative.Dark24)

    # Map base categories to colors
    base_color_map = {cat: colors[i % len(colors)] for i, cat in enumerate(unique_base_categories)}

    # Count occurrences of each category
    category_counts = Counter(categories_cleaned)

    # Create display labels with counts
    category_display = [f"{cat} ({category_counts[cat]})" for cat in categories_cleaned]

    # Create DataFrame
    df_plot = pd.DataFrame({
        'x': embedding[:, 0],
        'y': embedding[:, 1],
        'category_base': categories_cleaned,
        'category_display': category_display,
        'title': titles,
        'ai_material': ai_materials,
        'all_categories': all_categories,
        'title_wrapped': [t[:100] + '...' if len(t) > 100 else t for t in titles]
    })

    # Create empty figure
    fig = go.Figure()

    # Add scatter trace for each category
    for i, category in enumerate(unique_base_categories):
        mask = df_plot['category_base'] == category
        category_data = df_plot[mask]

        if len(category_data) > 0:
            count = category_counts[category]

            # Create custom data for hover
            customdata = []
            for _, row in category_data.iterrows():
                ai_status = "Yes" if row['ai_material'] else "No"
                all_cats = ", ".join(row['all_categories']) if isinstance(row['all_categories'], list) else str(row['all_categories'])
                customdata.append([row['category_display'], ai_status, all_cats, wrap_text(row['title'], width=80)])

            fig.add_trace(go.Scatter(
                x=category_data['x'],
                y=category_data['y'],
                mode='markers',
                marker=dict(
                    color=base_color_map[category],
                    size=5,
                    line=dict(width=1, color='white'),
                    opacity=0.8
                ),
                name=f"{category} ({count})",
                legendgroup=category,
                customdata=customdata,
                hovertemplate='<b>Case:</b><br>%{customdata[3]}<br><br>' +
                              '<b>Main Category:</b> %{customdata[0]}<br>' +
                              '<b>All Categories:</b> %{customdata[2]}<br>' +
                              '<b>AI Material:</b> %{customdata[1]}<br>' +
                              '<extra></extra>'
            ))

    # Update layout
    fig.update_layout(
        title={
            'text': 'Interactive Legal Cases UMAP Visualization<br><sub>Cases clustered by name similarity and categorized by content</sub>',
            'x': 0.5,
            'font': {'size': 20}
        },
        xaxis_title='UMAP Dimension 1',
        yaxis_title='UMAP Dimension 2',
        font=dict(size=12),
        legend=dict(
            title='Category (Count)',
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='black',
            borderwidth=1
        ),
        width=2800, # Increased width
        height=1800, # Increased height
        margin=dict(r=300),
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="black",
            font_size=12,
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
    fig.write_image(output_file) # Changed from write_html to write_image
    print(f"Static visualization saved to: {output_file}")

    return fig

def print_cluster_analysis(embedding, categories, titles, ai_materials):
    """Print basic analysis of the clusters"""
    print("=== CLUSTER ANALYSIS ===")
    print(f"Total cases: {len(categories)}")
    print(f"Embedding shape: {embedding.shape}")
    
    print("\nCategory distribution:")
    category_counts = Counter(categories)
    for category, count in category_counts.most_common():
        percentage = (count / len(categories)) * 100
        print(f"  {category}: {count} cases ({percentage:.1f}%)")
    
    print("\nAI Material distribution:")
    ai_count = sum(ai_materials)
    non_ai_count = len(ai_materials) - ai_count
    print(f"  AI Material: {ai_count} cases ({ai_count/len(ai_materials)*100:.1f}%)")
    print(f"  Non-AI Material: {non_ai_count} cases ({non_ai_count/len(ai_materials)*100:.1f}%)")
    
    print(f"\nUMAP embedding range:")
    print(f"  X: {embedding[:, 0].min():.2f} to {embedding[:, 0].max():.2f}")
    print(f"  Y: {embedding[:, 1].min():.2f} to {embedding[:, 1].max():.2f}")

def main():
    """Main function to run the complete interactive UMAP visualization pipeline"""
    print("Loading and processing data...")
    df, processed_cases = load_and_process_data()
    
    # Extract data for processing
    titles = [case['title'] for case in processed_cases]
    summaries = [case['summary'] for case in processed_cases]  # Same as titles in this format
    categories = [case['main_category'] for case in processed_cases]
    ai_materials = [case['ai_material'] for case in processed_cases]
    all_categories = [case['all_categories'] for case in processed_cases]
    
    print(f"Loaded {len(titles)} cases")
    
    # Create text features
    print("Creating TF-IDF features from case names...")
    features, vectorizer = create_text_features(summaries)
    print(f"Created {features.shape[1]} features")
    
    # Perform UMAP
    print("Performing UMAP dimensionality reduction...")
    embedding, reducer = perform_umap(features)

    # Evaluate UMAP with silhouette scores
    non_category_silhouette(features, embedding, categories)

    # Create interactive visualization
    print("Creating interactive visualization...")
    fig = create_interactive_visualization(embedding, categories, titles, ai_materials, all_categories)
    
    # Print analysis
    print_cluster_analysis(embedding, categories, titles, ai_materials)
    
    return embedding, categories, titles, fig

# Example usage
if __name__ == "__main__":
    try:
        embedding, categories, titles, fig = main()
        print("Interactive UMAP visualization completed successfully!")
        
    except FileNotFoundError:
        print(f"Error: Could not find the file '{DIR_PATH}'")
        print("Please make sure the file exists and the path is correct.")