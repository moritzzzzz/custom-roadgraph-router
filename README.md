# Custom Roadgraph Router

The **Custom Roadgraph Router** is a Python script designed to find a route on features of a Mapbox Tileset. It retrieves Mapbox vector tiles from a specified tileset and generate a routing graph using the NetworkX library. By providing origin and destination geolocations, the script calculates the shortest path between the two points and returns the route geometry. Additionally, it creates a visual representation of the roadgraph, displaying connected nodes and the route in context.

## Features

- Retrieve Mapbox vector tiles using a tileset identifier and access token.
- Generate a routing graph with NetworkX.
- Enhance the routing graph with additional logic for improved pathfinding.
- Calculate the shortest path from origin to destination.
- Visualize the roadgraph and route using Matplotlib.

## Dependencies

To run the Custom Roadgraph Router, you need to install the following dependencies:

```plaintext
fastapi
uvicorn[standard]
python-multipart
requests
networkx
matplotlib
shapely
mercantile
protobuf==3.20.1

```

Additionally, you will need to install the vector-tile-base library from GitHub:

```
git clone https://github.com/mapbox/vector-tile-base.git
```

## Installation
1. Clone the repository:
   
```
git clone https://github.com/moritzzzzz/custom-roadgraph-router
cd <repository-directory>
```
2. Install the dependencies:

You can install the required dependencies using pip. It is recommended to use a virtual environment.

```
pip install -r requirements.txt
```

3. Install vector-tile-base:

Follow the instructions provided in the vector-tile-base repository to set it up locally. 

## Usage

To use the Custom Roadgraph Router, you need to provide the following parameters:

- Origin: A tuple representing the latitude and longitude of the starting point (e.g., (lat, lon)).
- Destination: A tuple representing the latitude and longitude of the endpoint (e.g., (lat, lon)).
- Mapbox Access Token: Your Mapbox access token for authentication.
- Tileset Identifier: The identifier of the Mapbox tileset you want to use.
- Tileset Layer Name: The name of the layer in the tileset on which to perform routing.

 ** Please note that the router is requiring the `isOpen` property of a feature to be `true` so that it is considered as a valid routing edge!**

 The custom-roadgraph-router will return a router `computed_route.geojson` and a visualization of the roadgraph `route_graph.png` for troubleshooting purposes.

 ## Troubleshooting

If you encounter issues while running the script, consider the following:

- Ensure that your Mapbox access token is valid and has the necessary permissions to access the tileset
- Verify that the tileset identifier and layer name are correct.
- Check that the features of tileset that represent the edges have a property `isOpen` and this property must be set to `true`
- Check that all dependencies are installed properly.
- Review the generated roadgraph `road_graph.png` drawing for insights into the routing process.

Example symbolic road graph with origin, destination and route:

![image](https://github.com/user-attachments/assets/ac983d0e-20b8-4bb7-84a4-aaee5fa78163)



