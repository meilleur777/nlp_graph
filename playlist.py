import json
import random
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import numpy as np
import plotly.graph_objects as go # Ensure plotly.graph_objects is imported

# Set up data directory and limit number of playlists
DATA_DIR = Path('data')
TOP_K = 5  # Limit playlists for testing
FILE_LIMIT = 10  # Load only FILE_LIMIT JSON files instead of all
MAX_NODES_VISUALIZE = 500  # Limit nodes for visualization performance

# Load JSON Files Efficiently
files = sorted(DATA_DIR.glob('*.json'))[:FILE_LIMIT]
all_playlists = []

for slice_path in tqdm(files, desc="Processing slices"):
    with slice_path.open('r', encoding='utf-8') as f:
        slice_data = json.load(f)
    all_playlists.extend(slice_data.get('playlists', []))

# Sort and select top K playlists based on followers
top_playlists = sorted(all_playlists, key=lambda pl: pl.get('num_followers', 0), reverse=True)[:TOP_K]

# Validate extraction
if not top_playlists:
    print("No playlists extracted! Check data availability.")
else:
    print(f"Extracted {len(top_playlists)} playlists.")
    print("First playlist keys:", list(top_playlists[0].keys()))

# Construct Co-occurrence Graph
G = nx.Graph()

for pl in top_playlists:
    # Randomly sample tracks to limit processing
    track_ids = list({tr['track_uri'] for tr in pl['tracks']})
    track_ids = random.sample(track_ids, min(len(track_ids), 50))  # Process max 50 tracks per playlist

    for tr in pl['tracks']:
        track_id = tr['track_uri']
        if track_id in track_ids and not G.has_node(track_id):
            G.add_node(
                track_id,
                name=tr['track_name'],
                artist=tr['artist_name'],
                duration=tr['duration_ms'],
                album=tr['album_name']
            )

    # Add weighted edges for co-occurring tracks
    for u, v in zip(track_ids[:-1], track_ids[1:]):
        if G.has_edge(u, v):
            G[u][v]['weight'] += 1
        else:
            G.add_edge(u, v, weight=1)

print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

def filter_graph_for_visualization(G, max_nodes=MAX_NODES_VISUALIZE):
    """Filter graph to most important nodes for better visualization performance"""
    if G.number_of_nodes() <= max_nodes:
        return G
    
    print(f"Filtering graph from {G.number_of_nodes()} to {max_nodes} nodes for visualization...")
    
    # Get nodes by degree centrality (most connected nodes)
    centrality = nx.degree_centrality(G)
    top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
    top_node_ids = [node for node, _ in top_nodes]
    
    # Create subgraph with top nodes
    subgraph = G.subgraph(top_node_ids).copy()
    print(f"Filtered graph: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")
    
    return subgraph

def nx_to_plotly_optimized(G):
    """Optimized conversion from NetworkX to Plotly with edge weights as thickness"""
    
    # Use faster layout for larger graphs
    if G.number_of_nodes() > 200:
        pos = nx.spring_layout(G, k=1, iterations=20)  # Fewer iterations for speed
    else:
        pos = nx.kamada_kawai_layout(G)
    
    # Get edge weights and normalize them for thickness
    edge_weights = []
    edges_list = list(G.edges(data=True))
    
    for edge in edges_list:
        weight = edge[2].get('weight', 1)
        edge_weights.append(weight)
    
    # Normalize weights to reasonable line widths (0.5 to 4.0)
    if edge_weights:
        min_weight = min(edge_weights)
        max_weight = max(edge_weights)
        weight_range = max_weight - min_weight if max_weight > min_weight else 1
        
        normalized_weights = []
        for weight in edge_weights:
            normalized = 0.5 + ((weight - min_weight) / weight_range) * 3.5
            normalized_weights.append(normalized)
    else:
        normalized_weights = [1.0] * len(edges_list)
    
    # Build edge traces - separate trace for each edge to allow individual styling
    edge_traces = []
    
    for i, (edge, weight) in enumerate(zip(edges_list, normalized_weights)):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        edge_trace = {
            'type': 'scatter',
            'x': [x0, x1, None],
            'y': [y0, y1, None],
            'mode': 'lines',
            'line': {
                'width': weight,
                'color': 'rgba(125,125,125,0.6)'
            },
            'hoverinfo': 'text',
            'hovertext': f"Weight: {edge_weights[i]}",
            'showlegend': False,
            'name': f'edge_{i}'
        }
        edge_traces.append(edge_trace)
    
    # Get node positions
    nodes = list(G.nodes())
    node_x = [pos[node][0] for node in nodes]
    node_y = [pos[node][1] for node in nodes]
    
    # Create hover text with track info
    hover_texts = []
    for node in nodes:
        data = G.nodes[node]
        # Calculate weighted degree for this node
        weighted_degree = sum(G[node][neighbor].get('weight', 1) for neighbor in G[node])
        
        hover_text = f"<b>{data.get('name', 'Unknown')}</b><br>"
        hover_text += f"Artist: {data.get('artist', 'Unknown')}<br>"
        hover_text += f"Album: {data.get('album', 'Unknown')}<br>"
        hover_text += f"Connections: {G.degree(node)}<br>"
        hover_text += f"Weighted Degree: {weighted_degree}"
        hover_texts.append(hover_text)
    
    # Create node sizes based on weighted degree
    weighted_degrees = []
    for node in nodes:
        weighted_degree = sum(G[node][neighbor].get('weight', 1) for neighbor in G[node])
        weighted_degrees.append(weighted_degree)
    
    max_weighted_degree = max(weighted_degrees) if weighted_degrees else 1
    min_size, max_size = 8, 25
    node_sizes = [min_size + (degree / max_weighted_degree) * (max_size - min_size) 
                  for degree in weighted_degrees]
    
    # Create node trace
    node_trace = {
        'type': 'scatter',
        'x': node_x,
        'y': node_y,
        'mode': 'markers',
        'hoverinfo': 'text',
        'hovertext': hover_texts,
        'marker': {
            'size': node_sizes,
            'color': '#1DB954',
            'line': {'width': 1, 'color': 'white'},
            'opacity': 0.8
        },
        'showlegend': False,
        'name': 'nodes'
    }
    
    return edge_traces, node_trace, pos, nodes

def create_interactive_graph(G, save_filename=None):
    """Create interactive graph with edge weights and hover highlighting"""

    # Filter graph for better performance
    filtered_G = filter_graph_for_visualization(G)

    # Convert to Plotly format
    edge_traces, node_trace, pos, nodes = nx_to_plotly_optimized(filtered_G)

    # Create the figure
    fig = go.Figure()

    # Add all edge traces
    for edge_trace in edge_traces:
        fig.add_trace(go.Scatter(**edge_trace))

    # Add node trace
    fig.add_trace(go.Scatter(**node_trace))

    # Update layout (as you had it)
    fig.update_layout(
        title=dict(
            text=f"Interactive Spotify Network ({filtered_G.number_of_nodes()} tracks, {filtered_G.number_of_edges()} connections)",
            x=0.5,
            font=dict(size=16)
        ),
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=60),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white'
    )

    # Build node connections mapping for JavaScript
    node_connections = {}
    for i, node in enumerate(nodes):
        neighbors = list(filtered_G[node].keys())
        neighbor_indices = [nodes.index(neighbor) for neighbor in neighbors if neighbor in nodes]
        node_connections[i] = neighbor_indices

    # Create custom HTML with JavaScript for hover highlighting
    html_template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { margin: 0; padding: 20px; font-family: Arial, sans-serif; }
            #graph { width: 100%; height: 90vh; }
            .info { margin-bottom: 10px; font-size: 12px; color: #666; }
        </style>
    </head>
    <body>
        <div class="info">
            Edge thickness = co-occurrence strength | Node size = weighted connections | Hover over nodes to highlight neighbors
        </div>
        <div id="graph"></div>

        <script>
            // Graph data will be inserted here
            var graphData = GRAPH_DATA_PLACEHOLDER;
            var layout = LAYOUT_PLACEHOLDER;
            var nodeConnections = NODE_CONNECTIONS_PLACEHOLDER;

            // Plot the graph
            Plotly.newPlot('graph', graphData, layout, {displayModeBar: true, responsive: true});

            var graphDiv = document.getElementById('graph');
            var originalColors = [];
            var originalSizes = [];
            var isHighlighting = false;

            // Store original node properties
            var nodeTraceIndex = graphData.length - 1; // Last trace is nodes
            if (nodeTraceIndex >= 0 && graphData[nodeTraceIndex] && graphData[nodeTraceIndex].marker) {
                originalColors = [...graphData[nodeTraceIndex].marker.color];
                originalSizes = [...graphData[nodeTraceIndex].marker.size];
            } else {
                 console.warn("Could not find node trace or its marker properties for initial setup.");
            }

            // Hover event to highlight connected nodes
            graphDiv.on('plotly_hover', function(data) {
                if (isHighlighting) return;
                isHighlighting = true;

                var pointIndex = data.points[0].pointIndex;
                var traceIndex = data.points[0].curveNumber; // Use curveNumber for trace index

                // Only highlight if hovering over nodes (the node trace)
                if (traceIndex === nodeTraceIndex) {
                    var connectedNodes = nodeConnections[pointIndex] || [];
                    var newColors = [...originalColors];
                    var newSizes = [...originalSizes];

                    // Highlight connected nodes with mint green
                    connectedNodes.forEach(function(nodeIdx) {
                        newColors[nodeIdx] = '#00FF9F'; // Mint green
                        newSizes[nodeIdx] = originalSizes[nodeIdx] + 5;
                    });

                    // Update the node trace
                    Plotly.restyle('graph', {
                        'marker.color': [newColors],
                        'marker.size': [newSizes]
                    }, nodeTraceIndex);
                }

                setTimeout(function() { isHighlighting = false; }, 100);
            });

            // Unhover event to reset colors
            graphDiv.on('plotly_unhover', function(data) {
                if (data && data.points && data.points.length > 0 && data.points[0].curveNumber === nodeTraceIndex) {
                    Plotly.restyle('graph', {
                        'marker.color': [originalColors],
                        'marker.size': [originalSizes]
                    }, nodeTraceIndex);
                }
            });
        </script>
    </body>
    </html>
    '''

    # Save as HTML with custom interactivity
    if save_filename:
        try:
            html_filename = save_filename.replace('.svg', '.html')

            # Get the dictionary representation of the figure
            fig_dict = fig.to_dict()

            # Extract data and layout, which are now serializable dictionaries
            graph_data_json = json.dumps(fig_dict['data'])
            layout_json = json.dumps(fig_dict['layout'])
            node_connections_json = json.dumps(node_connections)

            # Create custom HTML with hover highlighting
            html_content = html_template
            html_content = html_content.replace('GRAPH_DATA_PLACEHOLDER', graph_data_json)
            html_content = html_content.replace('LAYOUT_PLACEHOLDER', layout_json)
            html_content = html_content.replace('NODE_CONNECTIONS_PLACEHOLDER', node_connections_json)

            with open(html_filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"Interactive graph saved to {html_filename}")

        except Exception as e:
            print(f"Error creating custom HTML with JavaScript: {e}")
            # Fallback to regular HTML save
            html_filename = save_filename.replace('.svg', '.html')
            fig.write_html(html_filename)
            print(f"Fallback: Interactive graph saved to {html_filename} using default write_html.")

    return fig

# Create and save the interactive graph
filename = f"graph_topK{TOP_K}_fileLimit{FILE_LIMIT}.svg"
fig = create_interactive_graph(G, save_filename=filename)