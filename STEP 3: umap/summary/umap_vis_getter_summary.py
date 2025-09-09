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

DIR_PATH = 'STEP 2: get embed/summaries/categorized_cases.json'
EMBEDDING_DIR = "STEP 2: get embed/summaries/embeddings/legalbert_embeddings.npy"
TEXT_TYPE = "summary"
MIN_DIST = 0.01
N_NEIGHBORS = 15
def load_and_process_data():
    """Load JSON data and process it for visualization (new JSON format)."""
    with open(DIR_PATH, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    processed_cases = []
    for count, (casename, metadata) in enumerate(data.items(), start=1):
        try:
            summary = metadata.get("summary", casename)
            categories_obj = metadata.get("categories", {})
            categories = categories_obj.get("category", ["Unknown"])
            ai_material = categories_obj.get("ai_material", False)
            main_category = categories[0] if isinstance(categories, list) and categories else "Unknown"
        except Exception as e:
            print(f"COUNT {count} – Warning: could not parse {casename}: {e}")
            summary = casename
            main_category = "Unknown"
            ai_material = False
            categories = ["Unknown"]

        case_info = {
            "title": casename,
            "summary": summary,
            "main_category": main_category,
            "ai_material": ai_material,
            "all_categories": categories,
        }
        processed_cases.append(case_info)
    
    df = pd.DataFrame(processed_cases)
    return df, processed_cases

def wrap_text(text, width=60):
    """Helper function to wrap text for better display"""
    return '<br>'.join(textwrap.wrap(str(text), width=width))

def create_interactive_visualization(embedding, categories, titles, ai_materials, all_categories, silhouette_scores, embedding_method="Legal-BERT"):
    """Create extra large UMAP visualization for presentations or large displays."""
    output_file=('umap_vis_' + embedding_method + '_' + TEXT_TYPE + '.png')
    # Clean categories
    categories_cleaned = [c.strip() if isinstance(c, str) else 'Unknown' for c in categories]

    # Get unique base categories and assign colors
    unique_base_categories = sorted(set(categories_cleaned))

    # Create color palette
    colors = (px.colors.qualitative.Bold +
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
                    size=12,  # Even larger markers
                    line=dict(width=2, color='white'),  # Thicker border
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

    # Update layout for presentation/large display
    fig.update_layout(
        title={
            'text': 'Legal Cases Visualization<br><sub>' + embedding_method + '</sub>',
            'x': 0.5,
            'font': {'size': 36}  # Very large title
        },
        xaxis_title=(embedding_method + ' Dimension 1'),
        yaxis_title=(embedding_method + ' Dimension 2'),
        xaxis=dict(
            title_font=dict(size=28),  # Large axis title font
            tickfont=dict(size=20)     # Large axis tick font
        ),
        yaxis=dict(
            title_font=dict(size=28),  # Large axis title font
            tickfont=dict(size=20)     # Large axis tick font
        ),
        font=dict(size=20),  # Large base font
        legend=dict(
            title='Category (Count)',
            title_font=dict(size=24),  # Large legend title
            font=dict(size=18),        # Large legend items
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='black',
            borderwidth=2
        ),
        # Extra large dimensions
        width=2560,   # 1440p width
        height=1440,  # 1440p height
        margin=dict(r=400, l=100, t=150, b=100),  # Larger margins
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="black",
            font_size=20,  # Larger hover font
            font_family="Arial",
            align="left"
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # Add the silhouette score annotation below the legend
     # Add all silhouette score annotations stacked under the legend
    y_offset = 0.02
    for i, (k, score) in enumerate(silhouette_scores.items()):
        fig.add_annotation(
            text=f"Silhouette (k={k}): {score:.4f}",
            xref="paper", yref="paper",
            x=1.02, y=y_offset + i*0.04,  # stack vertically
            showarrow=False,
            font=dict(size=16, color="black"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=2,
            borderpad=6,
            xanchor="left",
            yanchor="bottom"
        )
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    # Save with high DPI
    fig.write_image(output_file, width=2560, height=1440, scale=2)
    print(f"Extra large visualization saved to: {output_file}")

    return fig

def calculate_silhouette_scores(X, embedding_method, k_range, random_state=42):
    """Calculate silhouette scores for a range of k values"""
    print("Calculating silhouette scores for", embedding_method)
    scores = {}
    for k in k_range:
        if k < len(X):  # avoid invalid k
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            print("Silhouette score for k =", k, "is", score)
            scores[k] = score
    print("Highest sihouette score:", max(scores.values()), "for k =", max(scores, key=scores.get))
    return scores

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
    np.save('umap_embedding.npy', embedding)
    return embedding, reducer


def main():
    print("Loading and processing data...")
    df, processed_cases = load_and_process_data()  # Use the new loader

    # Extract data for processing
    titles = [case['title'] for case in processed_cases]
    summaries = [case['summary'] for case in processed_cases]
    categories = [case['main_category'] for case in processed_cases]
    ai_materials = [case['ai_material'] for case in processed_cases]
    all_categories = [case['all_categories'] for case in processed_cases]

    print(f"Loaded {len(titles)} cases")

    # Load embeddings
    embedding_path = EMBEDDING_DIR
    embedding_method = embedding_path.split('/')[-1].split('_')[0]
    embedding = np.load(embedding_path)

    # Apply UMAP
    embedding_umap, reducer = perform_umap(embedding)

    # Clustering + silhouette scores
    silhouette_scores = calculate_silhouette_scores(embedding_umap, embedding_method, range(3, 11))

    # Create visualization
    print("Creating interactive visualization...")
    fig = create_interactive_visualization(
        embedding_umap, categories, titles, ai_materials, all_categories, silhouette_scores
    )

    return embedding_umap, categories, titles, fig


# Example usage
if __name__ == "__main__":
    try:
        embedding_path = EMBEDDING_DIR
        text_type = TEXT_TYPE
        embedding = np.load(embedding_path)
        embedding, categories, titles, fig= main()
        print("Interactive visualization completed successfully!")
        
    except FileNotFoundError:
        print(f"Error: Could not find the file '{DIR_PATH}'")
        print("Please make sure the file exists and the path is correct.")