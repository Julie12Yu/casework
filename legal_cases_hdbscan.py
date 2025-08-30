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

DIR_PATH = 'all_jsons/categorized_cases.json'
MIN_CLUSTER_SIZE = 5
MIN_SAMPLES = 5

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

def perform_HDBSCAN(features):
    """Perform HDBSCAN dimensionality reduction"""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
    )
    reducer = None
    if IS_UMAP:
        clusterer.set_params(metric='euclidean', cluster_selection_method='eom')
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42) ### Make sure we make this slider later

    embeddings = reducer.fit_transform(features_scaled)
    clusterer = hdbscan.HDBSCAN(MIN_CLUSTER_SIZE, MIN_SAMPLES)
    clusterer.fit(embeddings)

    return embeddings

def wrap_text(text, width=60):
    """Helper function to wrap text for better display"""
    return '<br>'.join(textwrap.wrap(str(text), width=width))

def create_interactive_visualization(embedding, categories, titles, summaries, output_file='legal_cases_hdbscan_interactive.html'):
    """Create interactive HDBSCAN visualization with correct color mapping"""
    
    # Clean categories
    categories_cleaned = [c.strip() if isinstance(c, str) else 'Unknown' for c in categories]
    
    # Get unique base categories and assign colors
    unique_base_categories = sorted(set(categories_cleaned))
    
    # Create color palette
    colors = (px.colors.qualitative.Set3 + 
             px.colors.qualitative.Bold + 
             px.colors.qualitative.Dark24 + 
             px.colors.qualitative.Pastel)
    
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
        'summary': summaries,
        'summary_preview': [s[:300] + '...' if len(s) > 300 else s for s in summaries],
        'title_wrapped': [t[:80] + '...' if len(t) > 80 else t for t in titles]
    })
    
    # Wrap summary text
    df_plot['summary_wrapped'] = df_plot['summary_preview'].apply(lambda x: wrap_text(x, width=60))
    
    # Create empty figure
    fig = go.Figure()
    
    # Add scatter trace for each category to ensure consistent colors
    for i, category in enumerate(unique_base_categories):
        mask = df_plot['category_base'] == category
        category_data = df_plot[mask]
        
        if len(category_data) > 0:
            count = category_counts[category]
            
            fig.add_trace(go.Scatter(
                x=category_data['x'],
                y=category_data['y'],
                mode='markers',
                marker=dict(
                    color=base_color_map[category],
                    size=8,
                    line=dict(width=0.5, color='white')
                ),
                name=f"{category} ({count})",
                legendgroup=category,
                customdata=np.column_stack((
                    category_data['category_display'],
                    category_data['title_wrapped'],
                    category_data['summary_wrapped']
                )),
                hovertemplate='<b>%{customdata[1]}</b><br><br>' +
                              '<b>Category:</b> %{customdata[0]}<br><br>' +
                              '<b>Summary:</b><br>%{customdata[2]}<br>' +
                              '<extra></extra>'
            ))
    
    # Update layout
    title_text = 'Interactive Legal Cases HDBSCAN Visualization'
    if IS_UMAP:
        title_text += ' (with UMAP)'
    else:
        title_text += ' (with TSNE)'

    fig.update_layout(
        title={
            'text': title_text,
            'x': 0.5,
            'font': {'size': 24}
        },
        xaxis_title='HDBSCAN Component 1',
        yaxis_title='HDBSCAN Component 2',
        font=dict(size=12),
        legend=dict(
            title='Category (Count)',
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

def print_cluster_analysis(embedding, categories, titles):
    """Print basic analysis of the clusters"""
    print("=== CLUSTER ANALYSIS ===")
    print(f"Total cases: {len(categories)}")
    print(f"Embedding shape: {embedding.shape}")
    print("\nCategory distribution:")
    
    category_counts = Counter(categories)
    for category, count in category_counts.most_common():
        percentage = (count / len(categories)) * 100
        print(f"  {category}: {count} cases ({percentage:.1f}%)")
    
    print(f"\nHDBSCAN embedding range:")
    print(f"  X: {embedding[:, 0].min():.2f} to {embedding[:, 0].max():.2f}")
    print(f"  Y: {embedding[:, 1].min():.2f} to {embedding[:, 1].max():.2f}")

def main():
    """Main function to run the complete interactive HDBSCAN visualization pipeline"""
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
    
    # Perform HDBSCAN
    print("Performing HDBSCAN dimensionality reduction...")
    embedding = perform_HDBSCAN(features)
    
    # Create interactive visualization
    print("Creating interactive visualization...")
    fig = create_interactive_visualization(embedding, categories, titles, summaries)
    
    # Print analysis
    print_cluster_analysis(embedding, categories, titles)
    
    return embedding, categories, titles, fig

# Example usage
if __name__ == "__main__":    
    try:
        main()
        print("Interactive HDBSCAN visualization completed successfully!")
        
    except FileNotFoundError:
        print(f"Error: Could not find the file '{DIR_PATH}'")
        print("Please make sure the file exists and the path is correct.")