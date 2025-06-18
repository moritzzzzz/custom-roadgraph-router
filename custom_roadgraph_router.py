
#!/usr/bin/env python

import os
import json
import math
from itertools import combinations

import requests
import mercantile
import networkx as nx
from vector_tile_base import VectorTile
from shapely.geometry import Point, LineString, MultiLineString, mapping
from shapely.ops import unary_union
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
# Your Mapbox access token should be set as an environment variable.
MAPBOX_ACCESS_TOKEN = "YOUR_ACCESS_TOKEN"
tileset_identifier = "username.tileset_identifier"
# The layer name within your vector tileset that contains the trails.
# This must match the name of the source layer from your GeoJSON upload.
TRAIL_LAYER_NAME = 'TILESET_LAYER_NAME'  # <-- IMPORTANT: Change this if your layer has a different name.
# The property in your roadgraph features that indicates if a edge is open.
OPEN_PROPERTY_KEY = 'isOpen'
# The zoom level to use for fetching tiles. Higher zoom provides more detail
# but increases the number of tiles to download and process.
ROUTING_ZOOM_LEVEL = 14
# A buffer in degrees to add to the bounding box to ensure we capture
# trails that might be part of an optimal route but are slightly outside
# the direct bounding box of the origin and destination.
BBOX_BUFFER_DEG = 0.04
PROXIMITY_THRESHOLD_M = 50  # (in meters) to connect close nodes with artificial edges


def haversine_distance(coord1, coord2):
    """
    Calculate the great-circle distance between two points
    on the earth (specified in decimal degrees).
    Returns distance in meters.
    """
    R = 6372800  # Earth radius in meters
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2

    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def calculate_linestring_length_m(linestring):
    """Calculates the total geodesic length of a Shapely LineString in meters."""
    total_length = 0
    coords = list(linestring.coords)
    for i in range(len(coords) - 1):
        total_length += haversine_distance(coords[i], coords[i + 1])
    return total_length


def get_tiles_for_bbox(origin_lonlat, dest_lonlat, zoom, buffer_deg):
    """
    Calculates the list of Web Mercator tiles covering the bounding box
    of the origin and destination points, plus a buffer.
    """
    min_lon = min(origin_lonlat[0], dest_lonlat[0]) - buffer_deg
    max_lon = max(origin_lonlat[0], dest_lonlat[0]) + buffer_deg
    min_lat = min(origin_lonlat[1], dest_lonlat[1]) - buffer_deg
    max_lat = max(origin_lonlat[1], dest_lonlat[1]) + buffer_deg

    # Use mercantile to get the tiles for the bounding box at the specified zoom
    tiles = list(mercantile.tiles(min_lon, min_lat, max_lon, max_lat, zooms=[zoom]))
    return tiles


def fetch_tile_data(tileset_id, tile_x, tile_y, tile_z, access_token):
    """
    Fetches a single vector tile from the Mapbox Vector Tiles API.
    """
    api_url = f"https://api.mapbox.com/v4/{tileset_id}/{tile_z}/{tile_x}/{tile_y}.mvt?access_token={access_token}"
    try:
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Error fetching tile {tile_z}/{tile_x}/{tile_y}: {e}")
        return None


def parse_mvt_to_features(mvt_data, tile_x, tile_y, tile_z, layer_name, tile_extent=4096):
    """
    Parses binary MVT data, transforms tile-local coordinates to geographic
    coordinates (lon/lat), and filters for open trail features.
    """
    # Get the geographic bounds of the tile for coordinate transformation
    bounds = mercantile.bounds(tile_x, tile_y, tile_z)

    def transform_coords(points):
        """Helper function to transform a list of tile-local points to lon/lat."""
        transformed_points = []
        for point in points:
            # Simple linear interpolation from tile coords to lon/lat
            lon = bounds.west + (point[0] / tile_extent) * (bounds.east - bounds.west)
            lat = bounds.north - (point[1] / tile_extent) * (bounds.north - bounds.south)
            transformed_points.append((lon, lat))
        return transformed_points

    # Initialize the VectorTile object with the raw MVT data
    tile = VectorTile(mvt_data)

    trail_features = []
    for layer in tile.layers:
        if layer.name == layer_name:
            for feature in layer.features:
                # The feature type is an enum: 1=Point, 2=LineString, 3=Polygon
                # The attributes object does not have a.get() method, so we access it like a dictionary.
                if feature.type == "line_string" and feature.attributes[OPEN_PROPERTY_KEY] == "true":
                    # get_geometry() returns a list of line parts (for MultiLineString support) [2]
                    local_geometry = feature.get_geometry()
                    if local_geometry:
                        # We must loop through the list of line parts.
                        # A simple LineString will be a list with one element.
                        # A MultiLineString will have multiple elements.
                        for line_part in local_geometry:
                            if line_part: # ensure the line part is not empty
                                geo_coords = transform_coords(line_part)
                                if len(geo_coords) > 1: # A valid LineString needs at least 2 points
                                    line = LineString(geo_coords)
                                    trail_features.append({'geometry': line, 'properties': feature.attributes})
    return trail_features


def parse_mvt_to_features_old(mvt_data, tile_x, tile_y, tile_z, layer_name, tile_extent=4096):
    """
    Parses binary MVT data, transforms tile-local coordinates to geographic
    coordinates (lon/lat), and filters for open trail features.
    """
    # Get the geographic bounds of the tile for coordinate transformation
    bounds = mercantile.bounds(tile_x, tile_y, tile_z)

    def transform_coords(points):
        """Helper function to transform a list of tile-local points to lon/lat."""
        transformed_points = []
        for point in points:
            # Simple linear interpolation from tile coords to lon/lat
            lon = bounds.west + (point[0] / tile_extent) * (bounds.east - bounds.west)
            lat = bounds.north - (point[1] / tile_extent) * (bounds.north - bounds.south)
            transformed_points.append((lon, lat))
        return transformed_points

    # Initialize the VectorTile object with the raw MVT data
    tile = VectorTile(mvt_data)

    trail_features = []
    for layer in tile.layers:
        if layer.name == layer_name:
            for feature in layer.features:
                # Filter for LineString features with the "isOpen" property set to True
                # The feature type is an enum: 1=Point, 2=LineString, 3=Polygon
                if feature.type == "line_string" and feature.attributes[OPEN_PROPERTY_KEY] == "true":
                    # Get tile-local geometry [2]
                    local_geometry = feature.get_geometry()

                    # Assuming the trail is a single LineString, not a MultiLineString
                    if local_geometry and local_geometry:
                        # Transform the coordinates of the first (and only) line part
                        geo_coords = transform_coords(local_geometry)

                        # Create a Shapely LineString object
                        line = LineString(geo_coords)
                        trail_features.append({'geometry': line, 'properties': feature.attributes})
    return trail_features


def build_graph_from_features(trail_features):
    """
    Constructs a NetworkX graph from a list of Shapely LineString features.
    It correctly nodes the network at all intersections.
    """
    print("  Merging trail segments...")
    # Use unary_union to merge all line segments and correctly node them at intersections
    all_geometries = [f['geometry'] for f in trail_features]
    merged_lines = unary_union(all_geometries)

    graph = nx.Graph()

    # The result of unary_union can be a single LineString or a MultiLineString
    if merged_lines.geom_type == 'LineString':
        lines_to_process = [merged_lines]
    elif merged_lines.geom_type == 'MultiLineString':
        lines_to_process = list(merged_lines.geoms)
    else:
        lines_to_process = []

    print(f"  Adding {len(lines_to_process)} segments to the graph...")
    for segment in lines_to_process:
        # Each line in the merged geometry is an edge in the graph.
        # Ensure nodes are the coordinate tuples, not CoordinateSequence objects.
        start_node = segment.coords[0]
        end_node = segment.coords[-1]

        # Calculate the geodesic length in meters for the edge weight
        length_m = calculate_linestring_length_m(segment)

        # Add an edge to the graph with its length as the weight
        # and the geometry stored as an attribute.
        graph.add_edge(
            start_node,
            end_node,
            weight=length_m,
            geometry=segment
        )

    return graph


def add_proximity_edges(graph, threshold_m):
    """
    Adds edges between nodes that are closer than the threshold but not yet connected.
    """
    print(f"\nStep 2.5: Connecting nearby nodes within {threshold_m}m...")
    nodes = list(graph.nodes())
    added_edges = 0
    # Use itertools.combinations to get all unique pairs of nodes
    for u, v in combinations(nodes, 2):
        # Check if an edge already exists
        if not graph.has_edge(u, v):
            # Calculate distance in meters
            distance_m = haversine_distance(u, v)
            if distance_m < threshold_m:
                # Add a new edge if distance is below threshold
                graph.add_edge(
                    u,
                    v,
                    weight=distance_m,  # Use the actual distance as weight
                    geometry=LineString([u, v])  # Create a straight line geometry
                )
                added_edges += 1
    print(f"  Added {added_edges} new proximity edges.")
    return graph


def find_shortest_route(graph, origin_lonlat, dest_lonlat):
    """
    Finds the shortest route in the graph between an origin and destination.
    """
    origin_point = Point(origin_lonlat)
    dest_point = Point(dest_lonlat)

    # Find the graph nodes closest to the origin and destination points
    print("  Finding nearest trail access points...")
    nodes = list(graph.nodes)

    # Calculate distances from origin to all nodes
    origin_distances = [origin_point.distance(Point(node)) for node in nodes]
    start_node_idx = min(range(len(origin_distances)), key=origin_distances.__getitem__)
    start_node = nodes[start_node_idx]

    # Calculate distances from destination to all nodes
    dest_distances = [dest_point.distance(Point(node)) for node in nodes]
    end_node_idx = min(range(len(dest_distances)), key=dest_distances.__getitem__)
    end_node = nodes[end_node_idx]

    print(f"  Start node: {start_node}")
    print(f"  End node: {end_node}")

    # Use NetworkX's Dijkstra algorithm to find the shortest path
    print("  Calculating shortest path...")
    try:
        path_nodes = nx.shortest_path(graph, source=start_node, target=end_node, weight='None')
        return path_nodes, start_node, end_node
    except nx.NetworkXNoPath:
        print("  Could not find a path between the start and end nodes.")
        return None, start_node, end_node


def create_route_geojson(path_nodes, graph):
    """
    Reconstructs the route's geometry from the path nodes and the graph,
    and formats it as a GeoJSON Feature.
    """
    if not path_nodes or len(path_nodes) < 2:
        return None

    route_segments = []
    total_length = 0
    for i in range(len(path_nodes) - 1):
        u = path_nodes[i]
        v = path_nodes[i + 1]
        edge_data = graph.get_edge_data(u, v)
        route_segments.append(edge_data['geometry'])
        total_length += edge_data['weight']

    # Combine the individual LineString segments into one
    # The unary_union is a clean way to merge contiguous linestrings
    full_route_geom = unary_union(route_segments)

    # Create the GeoJSON Feature
    route_geojson = {
        "type": "Feature",
        "geometry": mapping(full_route_geom),
        "properties": {
            "total_length_degrees": total_length,
            "note": "Length is in decimal degrees. For meters, reproject coordinates."
        }
    }
    return route_geojson


def draw_route_on_graph(graph, path_nodes, start_node, end_node, filename="route_graph.png"):
    """
    Draws the network graph. Each connected component is colored differently.
    If a route is provided, it is highlighted. The start and end nodes are always highlighted.
    Saves the output to a file.
    """
    print(f"\nStep 4: Drawing graph and saving to {filename}...")

    # Define positions for drawing. The node itself is the (lon, lat) tuple.
    pos = {node: node for node in graph.nodes()}

    plt.figure(figsize=(12, 12))

    # Find connected components and assign a color to each [3]
    components = nx.connected_components(graph)
    component_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                        '#17becf']

    # Draw nodes and edges for each component with a unique color
    for i, component in enumerate(components):
        color = component_colors[i % len(component_colors)]
        subgraph = graph.subgraph(component)
        nx.draw_networkx_nodes(graph, pos, nodelist=list(component), node_size=15, node_color=color)
        nx.draw_networkx_edges(graph, pos, edgelist=subgraph.edges(), edge_color=color)

    # If a path was found, highlight it in red on top
    if path_nodes:
        path_edges = list(zip(path_nodes, path_nodes[1:]))

        # Highlight the nodes and edges of the shortest path
        nx.draw_networkx_nodes(graph, pos, nodelist=path_nodes, node_size=25, node_color='red')
        nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color='red', width=2)

        plt.title("Trail Network and Calculated Route")
    else:
        plt.title("Trail Network (Connected Components Colored)")

    # Always highlight the start and end nodes on top
    if start_node:
        nx.draw_networkx_nodes(graph, pos, nodelist=[start_node], node_size=60, node_color='lime', edgecolors='black')
    if end_node:
        nx.draw_networkx_nodes(graph, pos, nodelist=[end_node], node_size=60, node_color='cyan', edgecolors='black')

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.grid(True)
    plt.axis('equal')  # Ensure aspect ratio is correct for geographic data

    try:
        plt.savefig(filename, bbox_inches='tight')
        print(f"Graph image saved successfully to {filename}")
    except Exception as e:
        print(f"Error saving graph image: {e}")


def main():
    """
    Main execution function for the routing engine.
    """
    if not MAPBOX_ACCESS_TOKEN:
        print("Error: MAPBOX_ACCESS_TOKEN environment variable not set.")
        return

    # --- USER INPUT ---
    # Replace with your actual origin, destination, and tileset ID.
    origin_point = (-115.9058690071106, 49.78126405817835)  # lon, lat
    destination_point = (-115.88975429534912, 49.76700478663415)

    print(f"Starting routing process...")
    print(f"From: {origin_point}")
    print(f"To:   {destination_point}")
    print(f"Tileset: {tileset_identifier}\n")

    if "your_username.your_tileset_id" in tileset_identifier:
        print("Warning: Please replace 'your_username.your_tileset_id' with your actual Mapbox tileset ID.")
        return

    # 1. Discover and fetch tiles
    print("Step 1: Discovering and fetching vector tiles...")
    tiles_to_fetch = get_tiles_for_bbox(origin_point, destination_point, ROUTING_ZOOM_LEVEL, BBOX_BUFFER_DEG)
    print(f"Found {len(tiles_to_fetch)} tiles to fetch at zoom {ROUTING_ZOOM_LEVEL}.")

    all_trail_features = []
    for tile in tiles_to_fetch:
        print(f"  Fetching tile: Z{tile.z}-X{tile.x}-Y{tile.y}")
        mvt_data = fetch_tile_data(tileset_identifier, tile.x, tile.y, tile.z, MAPBOX_ACCESS_TOKEN)
        if mvt_data:
            features = parse_mvt_to_features(mvt_data, tile.x, tile.y, tile.z, TRAIL_LAYER_NAME)
            all_trail_features.extend(features)
            print(f"    -> Found {len(features)} open trail features.")

    if not all_trail_features:
        print("\nNo open trail features found in the specified region. Cannot compute a route.")
        return

    # 2. Build graph
    print("\nStep 2: Building routing graph from trail features...")
    trail_graph = build_graph_from_features(all_trail_features)
    print(f"Graph built with {trail_graph.number_of_nodes()} nodes and {trail_graph.number_of_edges()} edges.")

    if not trail_graph.nodes:
        print("Graph could not be built. No nodes found.")
        return

    # 2.5 Add edges between nearby nodes
    trail_graph = add_proximity_edges(trail_graph, threshold_m=PROXIMITY_THRESHOLD_M)


    # 3. Find shortest route
    print("\nStep 3: Finding the shortest route...")
    path_nodes, start_node, end_node = find_shortest_route(trail_graph, origin_point, destination_point)


    if not path_nodes:
        print("\nFailed to find a route.")


    # 3.1. Draw the graph with the route highlighted
    draw_route_on_graph(trail_graph, path_nodes, start_node, end_node)


    # 4. Create route GeoJSON
    print("\nStep 4: Reconstructing route geometry...")
    route_geojson = create_route_geojson(path_nodes, trail_graph)

    if route_geojson:
        print("\n--- ROUTE FOUND ---")
        print("Computed Route GeoJSON:")
        print(json.dumps(route_geojson, indent=2))

        # Optionally, save to a file
        try:
            with open("computed_route.geojson", "w") as f:
                json.dump(route_geojson, f, indent=2)
            print("\nRoute saved to computed_route.geojson")
        except IOError as e:
            print(f"\nCould not save file: {e}")
    else:
        print("\nFailed to generate final route geometry.")


if __name__ == '__main__':
    main()
