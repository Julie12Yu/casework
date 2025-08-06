import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from collections import Counter
import textwrap

def load_and_process_data(json_file_path):
    """Load JSON data and process it for t-SNE visualization"""
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    df = pd.DataFrame(data)
    
    # Extract categories
    categories = []
    for case in data:
        if isinstance(case['category']['category'], list) and len(case['category']['category']) > 0:
            categories.append(case['category']['category'][0])
        else:
            categories.append('Unknown')
    
    df['main_category'] = categories
    return df, data

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

def perform_tsne(features, perplexity=30, n_components=2, random_state=42):
    """Perform t-SNE dimensionality reduction"""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    tsne = TSNE(
        n_components=n_components,
        perplexity=min(perplexity, len(features)-1),
        random_state=random_state,
        init='pca',
        n_iter=1000,
        learning_rate='auto'
    )
    
    embedding = tsne.fit_transform(features_scaled)
    return embedding, tsne

def wrap_text(text, width=60):
    """Helper function to wrap text for better display"""
    return '<br>'.join(textwrap.wrap(str(text), width=width))

def create_interactive_visualization(embedding, categories, titles, summaries, output_file='legal_cases_tsne_interactive.html'):
    """Create interactive t-SNE visualization with correct color mapping"""
    
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
    fig.update_layout(
        title={
            'text': 'Interactive Legal Cases t-SNE Visualization',
            'x': 0.5,
            'font': {'size': 24}
        },
        xaxis_title='t-SNE Component 1',
        yaxis_title='t-SNE Component 2',
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
    
    print(f"\nt-SNE embedding range:")
    print(f"  X: {embedding[:, 0].min():.2f} to {embedding[:, 0].max():.2f}")
    print(f"  Y: {embedding[:, 1].min():.2f} to {embedding[:, 1].max():.2f}")

def main(json_file_path, perplexity=30):
    """Main function to run the complete interactive t-SNE visualization pipeline"""
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
    
    # Perform t-SNE
    print("Performing t-SNE dimensionality reduction...")
    embedding, tsne_model = perform_tsne(features, perplexity=perplexity)
    
    # Create interactive visualization
    print("Creating interactive visualization...")
    fig = create_interactive_visualization(embedding, categories, titles, summaries)
    
    # Print analysis
    print_cluster_analysis(embedding, categories, titles)
    
    return embedding, categories, titles, fig

# Example usage
if __name__ == "__main__":
    json_file_path = 'classify_results.json'
    
    try:
        embedding, categories, titles, fig = main(
            json_file_path=json_file_path,
            perplexity=10  # Better for smaller datasets
        )
        print("Interactive t-SNE visualization completed successfully!")
        
    except FileNotFoundError:
        print(f"Error: Could not find the file '{json_file_path}'")
        print("Please make sure the file exists and the path is correct.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Required packages:
# pip install numpy pandas plotly scikit-learn