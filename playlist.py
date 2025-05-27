#!/usr/bin/env python
# coding: utf-8

import json
import random
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

print(plt.get_backend())

# Set up data directory and limit number of playlists
DATA_DIR = Path('data')
TOP_K = 1000  # Limit playlists for testing
FILE_LIMIT = 100  # Load only FILE_LIMIT JSON files instead of all

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

# Compute Degree Centrality
deg_cent = nx.degree_centrality(G)
top_deg = sorted(deg_cent.items(), key=lambda x: x[1], reverse=True)[:10]
print("\nTop 10 tracks by degree centrality:")
for node, c in top_deg:
    data = G.nodes[node]
    print(f"{data['name']} — {data['artist']} (centrality={c:.4f})")

# Playlist Recommendation Function
def build_playlist(G, seed_uri, length=10):
    playlist = [seed_uri]
    scores = {nbr: G[seed_uri][nbr]['weight'] for nbr in G[seed_uri]}
    while len(playlist) < length and scores:
        candidates = {t: s for t, s in scores.items() if t not in playlist}
        if not candidates:
            break
        next_track = max(candidates, key=candidates.get)
        playlist.append(next_track)
        for nbr, data in G[next_track].items():
            if nbr not in playlist:
                scores[nbr] = scores.get(nbr, 0) + data['weight']
    return playlist

name_map = {(data['name'].lower(), data['artist'].lower()): uri for uri, data in G.nodes(data=True)}

def recommend_playlist_by_name(seed_name, seed_artist, length=10):
    key = (seed_name.lower(), seed_artist.lower())
    if key not in name_map:
        print(f"Track not found: {seed_name} — {seed_artist}")
        return
    seed_uri = name_map[key]
    recommended = build_playlist(G, seed_uri, length)
    print(f"\n=== Recommended Playlist (Seed: {seed_name} — {seed_artist}) ===")
    for i, uri in enumerate(recommended, 1):
        d = G.nodes[uri]
        print(f"{i}. {d['name']} — {d['artist']}")

# Example Recommendations
# recommend_playlist_by_name("Thinking Out Loud", "Ed Sheeran", length=10)
# recommend_playlist_by_name("Love Yourself", "Justin Bieber", length=10)

# Visualize Graph

# Ensure the graph exists
if G.number_of_nodes() == 0:
    print("Graph is empty, skipping visualization.")
else:
    # Limit nodes to the top 500 by degree centrality
    top_nodes = [node for node, _ in sorted(nx.degree_centrality(G).items(), key=lambda x: x[1], reverse=True)[:500]]
    subgraph = G.subgraph(top_nodes)

    # Choose the layout (spring_layout or kamada_kawai_layout, spring_layout takes longer time)
    # pos = nx.spring_layout(subgraph, seed=42)
    pos = nx.kamada_kawai_layout(subgraph)

    # Filter edges to show only strong connections according to weight
    filtered_edges = [(u, v) for u, v, d in subgraph.edges(data=True) if d['weight'] >=0]

    # Graph Visualization
    plt.figure(figsize=(12, 12))
    nx.draw_networkx_nodes(subgraph, pos, node_size=50, alpha=0.8, node_color="#1DB954")
    nx.draw_networkx_edges(subgraph, pos, edgelist=filtered_edges, alpha=1)
    
    # Optional: Show track names (shortened for readability)
    labels = {node: data['name'][:15] for node, data in subgraph.nodes(data=True)}
    nx.draw_networkx_labels(subgraph, pos, labels, font_size=8)

    plt.title("Track Co-occurrence Graph", fontsize=14)
    plt.show()
