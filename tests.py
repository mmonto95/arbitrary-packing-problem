import json
from shapely.geometry import shape, box, Polygon, Point
from shapely.affinity import scale

with open('shapes/random_polygons.json', 'r') as f:
    shapes = json.load(f)
    shapes = [shape(s) for s in shapes]

with open('shapes/random_polygons_large.json', 'r') as f:
    shapes_large = json.load(f)
    shapes_large = [shape(s) for s in shapes_large]

radius = 0.125

containers = [
    box(-0.5, -0.5, 1, 0.5),
    box(-2, -2, 4, 2),
    box(-5, -5, 10, 5),
]

hole_containers = [
    containers[0].symmetric_difference(Point((0.25, 0)).buffer(0.3)),
    containers[1].symmetric_difference(Point((1, 0)).buffer(1.2)),
    containers[2].symmetric_difference(Point((4, 0)).buffer(4.8)),
]

polygon = Polygon(((-0.5, -0.5), (0.2, 0.2), (0.8, 0.2), (0.6, 0.6), (1, 0.6), (1, 0.7),
                   (0, 0.7), (-0.5, 0.5), (-0.2, 0)))

irregular_containers = [
    polygon,
    scale(polygon, 4, 4),
    scale(polygon, 16, 16),
]

irregular_hole_containers = [
    irregular_containers[0].symmetric_difference(Point((0.25, 0.45)).buffer(0.2)),
    irregular_containers[1].symmetric_difference(Point((1, 1.5)).buffer(0.6)),
    irregular_containers[2].symmetric_difference(Point((-3, 5)).buffer(3)),
]

ALL_TESTS = [
    {
        'name': 'rect_small',
        'container': containers[0],
        'shapes': shapes,
        'radius': radius,
    },
    {
        'name': 'rect_medium',
        'container': containers[1],
        'shapes': shapes,
        'radius': radius,
    },
    # {
    #     'name': 'rect_large',
    #     'container': containers[2],
    #     'shapes': shapes_large,
    #     'radius': radius,
    # },
    {
        'name': 'rect_hole_small',
        'container': hole_containers[0],
        'shapes': shapes,
        'radius': radius,
    },
    {
        'name': 'rect_hole_medium',
        'container': hole_containers[1],
        'shapes': shapes,
        'radius': radius,
    },
    # {
    #     'name': 'rect_hole_large',
    #     'container': hole_containers[2],
    #     'shapes': shapes_large,
    #     'radius': radius,
    # },
    {
        'name': 'irregular_small',
        'container': irregular_containers[0],
        'shapes': shapes,
        'radius': radius,
    },
    {
        'name': 'irregular_medium',
        'container': irregular_containers[1],
        'shapes': shapes,
        'radius': radius,
    },
    # {
    #     'name': 'irregular_large',
    #     'container': irregular_containers[2],
    #     'shapes': shapes_large,
    #     'radius': radius,
    # },
    {
        'name': 'irregular_hole_small',
        'container': irregular_hole_containers[0],
        'shapes': shapes,
        'radius': radius,
    },
    {
        'name': 'irregular_hole_medium',
        'container': irregular_hole_containers[1],
        'shapes': shapes,
        'radius': radius,
    },
    # {
    #     'name': 'irregular_hole_large',
    #     'container': irregular_hole_containers[2],
    #     'shapes': shapes_large,
    #     'radius': radius,
    # },
]

CONSTRUCTIVE_SEARCH = {
    'intersection_threshold': [40, 60, 80],
    'max_iter': [1000, 10000],
    'shots': [1, 3, 5],
    'n_neighbors': [5, 8, 12, 16],
    'step_portion': [2, 8, 16, 32],
    'queue_length': [3, 10, 20],
    'n_rotations': [1, 16, 32],
}

LOCAL_SEARCH = {
    **CONSTRUCTIVE_SEARCH,
    'n_search': [10, 100, 1000],
}

PSO_SEARCH = {
    'phi_p': [1, 2, 3],
    'phi_g': [1, 2, 3],
    'w': [0.1, 0.3, 0.5, 0.8, 1],
    'n_particles': [10, 100, 500],
    'n_iterations': [10, 100, 1000],
    'n_circle_iter': [3, 8, 15],
}

GRID_SEARCH = {
    **LOCAL_SEARCH,
    **PSO_SEARCH,
}
