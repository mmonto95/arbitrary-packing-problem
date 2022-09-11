import numpy as np
import pandas as pd
from shapely.geometry import shape, Point
from scipy.spatial.distance import squareform, pdist


class PolygonFiller:
    """
    This class takes polygon coordinates and returns coordinates of circles
    filling the given polygon
    """
    def __init__(self, geometry, radius, intersection_threshold=60, max_iter=1000,
                 shots=1, n_neighbors=5):
        """
        Args:
            geometry: Leaflet geometry
            radius: Circle radius in kilometers
            intersection_threshold: Exclude initial circles when their percentage of intersection with the polygon is\
             less than threshold
            max_iter: Max number of circle movement iterations
            shots: Number of algorithm runs (Useful for refining)
            n_neighbors: Number of neighbors to consider in case shots > 1
        """

        self.polygon = shape(geometry)
        self.coords = np.array(geometry['coordinates'][0])
        self.radius = radius
        self.diameter = self.radius * 2
        self.intersection_threshold = intersection_threshold
        self.max_iter = max_iter  # ToDo: Offer calculation based on polygon and circles size

        if shots < 1:
            raise ValueError('Number of shots must be >= 1')

        self.shots = shots  # ToDo: Offer calculation based on metric
        self.n_neighbors = n_neighbors

    @staticmethod
    def calculate_intersection(polygon, x, y, radius):
        circle = Point((x, y)).buffer(radius)
        return polygon.intersection(circle).area / circle.area * 100

    def get_c_area(self, df, x, y, neighbors, radius):
        c_area = 0
        for neighbor in neighbors:
            circle = Point((df.loc[neighbor, 'x'], df.loc[neighbor, 'y'])).buffer(radius)
            c_area += self.calculate_intersection(circle, x, y, radius)

        return c_area

    def get_intersections(self, df, x, y, neighbors, radius):
        p_area = 100 - self.calculate_intersection(self.polygon, x, y, radius)
        c_area = self.get_c_area(df, x, y, neighbors, radius)

        return p_area, c_area

    @staticmethod
    def get_circle_coords(x, y):
        circle_coords_list = []
        for i in x:
            for j in y:
                circle_coords_list.append([i, j])

        return circle_coords_list

    # noinspection PyArgumentList
    def initialize_circles(self):
        x_max, y_max = self.coords.max(axis=0)
        x_min, y_min = self.coords.min(axis=0)

        n_circles_x = (x_max - x_min) / self.diameter
        n_circles_y = (y_max - y_min) / self.diameter

        x_max = x_min + self.diameter * (n_circles_x + 1)
        y_max = y_min + self.diameter * (n_circles_y + 1)
        n_circles_x = round((x_max - x_min) / self.diameter)
        n_circles_y = round((y_max - y_min) / self.diameter)

        x = np.linspace(x_min, x_max, n_circles_x + 1)
        y = np.linspace(y_min, y_max, n_circles_y + 1)
        return self.get_circle_coords(x, y)

    def select_circles(self, circle_coords_list):
        selected_circles = []
        intersections = []
        for circle_coords in circle_coords_list:
            intersection = self.calculate_intersection(self.polygon, circle_coords[0], circle_coords[1], self.radius)
            if intersection >= self.intersection_threshold:
                intersections.append(intersection)
                selected_circles.append(circle_coords)

        return selected_circles

    def initialize_neighbors(self, selected_circles):
        df = pd.DataFrame(selected_circles, columns=['x', 'y'])

        df['neighbors'] = None
        # df['is_edge'] = False  # ToDo: See if this is useful
        # df['is_corner'] = False  # ToDo: See if this is useful
        for i, row in df.iterrows():
            df_y = df[df['y'] == row['y']].copy()
            df_x = df[df['x'] == row['x']].copy()
            df_y['order'] = (df_y['x'] - row['x']).abs().round(6)
            df_x['order'] = (df_x['y'] - row['y']).abs().round(6)
            neighbors = []
            df_y['order'] = df_y['order'].rank(method='dense')
            neighbors.extend(df_y[df_y['order'] == 2].index.tolist())
            df_x['order'] = df_x['order'].rank(method='dense')
            neighbors.extend(df_x[df_x['order'] == 2].index.tolist())
            df.at[i, 'neighbors'] = neighbors
            # if len(neighbors) == 2:
            #     df.loc[i, 'is_edge'] = True
            #     df.loc[i, 'is_corner'] = True
            # elif len(neighbors) == 3:
            #     df.loc[i, 'is_edge'] = True

        df['p_area'] = 0.  # Area of the circle outside the polygon
        df['c_area'] = 0.  # Area of the circle intersected with other circles
        for i, row in df.iterrows():
            p_area, c_area = self.get_intersections(df, row['x'], row['y'], row['neighbors'], self.radius)

            df.loc[i, 'p_area'] = p_area
            df.loc[i, 'c_area'] = c_area

        df['wrong_area'] = df['p_area'] + df['c_area']
        return df

    def calculate_neighbors(self, df):
        distances = pd.DataFrame(squareform(pdist(df.loc[:, ['y', 'x']])))
        distances.index = df.index
        distances.columns = df.index
        for idx, row in df.iterrows():
            df.at[idx, 'neighbors'] = distances.loc[idx].sort_values().head(self.n_neighbors + 1) \
                                          .index[1:].tolist()

        return df

    def optimize(self, df):
        step = self.radius / 2  # ToDo: add this as an optional parameter
        for shot in range(self.shots):

            if shot > 0:
                df = self.calculate_neighbors(df)

            # ToDo: Add circles not moving check (If many circles are not moving in consecutive iterations, break)
            df['locked'] = False
            queue = [-1] * 3
            for _ in range(self.max_iter):
                df_aux = df[~df['locked']].copy()
                if df_aux.empty:
                    break

                idx = df_aux['wrong_area'].idxmax()
                queue.append(idx)
                queue.pop(0)

                if queue == [idx, idx, idx]:  # If the last 3 iterations have moved the same circle, lock it
                    df.loc[idx, 'locked'] = True

                row = df.loc[idx]

                y = row['y']
                x = row['x']
                up = x, y + step
                right = x + step, y
                down = x, y - step
                left = x - step, y

                wrong_area_min = row['wrong_area']
                if wrong_area_min < 1e-5:  # ToDo: Add this as a parameter
                    break

                p_area_min = row['p_area']
                c_area_min = row['c_area']
                for direction in (up, right, down, left):
                    p_area, c_area = self.get_intersections(df, direction[0], direction[1], row['neighbors'],
                                                            self.radius)
                    if p_area + c_area < wrong_area_min:
                        p_area_min = p_area
                        c_area_min = c_area
                        wrong_area_min = p_area + c_area
                        x = direction[0]
                        y = direction[1]

                df.loc[idx, 'y'] = y
                df.loc[idx, 'x'] = x
                df.loc[idx, 'p_area'] = p_area_min
                df.loc[idx, 'c_area'] = c_area_min
                df.loc[idx, 'wrong_area'] = wrong_area_min

                for neighbor in row['neighbors']:
                    c_area = self.get_c_area(df, df.loc[neighbor, 'x'], df.loc[neighbor, 'y'],
                                             df.loc[neighbor, 'neighbors'], self.radius)
                    df.loc[neighbor, 'c_area'] = c_area
                    df.loc[neighbor, 'wrong_area'] = df.loc[neighbor, 'c_area'] + df.loc[neighbor, 'p_area']

        return df

    def fill(self):
        circle_coords_list = self.initialize_circles()
        selected_circles = self.select_circles(circle_coords_list)
        df = self.initialize_neighbors(selected_circles)
        return self.optimize(df)


class BeeInitialization:
    radius = None

    def get_circle_coords(self, x, y):
        circle_coords_list = []
        for i in x:
            for c, j in enumerate(y):
                if c % 2 == 0:
                    circle_coords_list.append([i, j])
                else:
                    circle_coords_list.append([i + self.radius, j])

        return circle_coords_list


class PolygonFillerBeeInitialization(BeeInitialization, PolygonFiller):
    pass


class PolygonFillerStrict(PolygonFiller):
    """
    Variant to PolygonFiller in which overlapping circles and circles with parts outside
    the polygon are not allowed
    """
    def optimize(self, df):
        while True:
            df = super().optimize(df)
            df = self.calculate_neighbors(df)
            df_wrong = df[df['wrong_area'] > 1e-10]  # ToDo: Add this as a parameter
            if len(df_wrong) > 0:
                # ToDo: Improve this computationally
                idx = df_wrong.sort_values('wrong_area', ascending=False).index[0]
                df.drop(idx, inplace=True)
                df['neighbors'] = df['neighbors'].apply(lambda l: [i for i in l if i != idx])
                # df = self.calculate_neighbors(df[['x', 'y']])
            else:
                break
        return df


class PolygonFillerBeeStrict(BeeInitialization, PolygonFillerStrict):
    pass



