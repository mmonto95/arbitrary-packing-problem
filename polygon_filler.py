import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist
from shapely.geometry import shape, Point
from shapely.affinity import rotate
from shapely.ops import unary_union


# TODO: Refactor this class in terms of the IrregularPacker
class PolygonFiller:
    """
    This class takes polygon coordinates and returns coordinates of circles
    filling the given polygon
    """
    def __init__(self, geometry, radius, intersection_threshold=60, max_iter=1000,
                 shots=1, n_neighbors=5):
        """
        Args:
            geometry: Shapely geometry
            radius: Circle radius in kilometers
            intersection_threshold: Exclude initial circles when their percentage of intersection with the polygon is\
             less than threshold
            max_iter: Max number of circle movement iterations
            shots: Number of algorithm runs (Useful for refining)
            n_neighbors: Number of neighbors to consider in case shots > 1
        """

        self.polygon = geometry
        self.coords = np.array(self.polygon.exterior.coords)
        self.radius = radius
        self.diameter = self.radius * 2
        self.intersection_threshold = intersection_threshold
        self.max_iter = max_iter  # TODO: Offer calculation based on polygon and circles size

        if shots < 1:
            raise ValueError('Number of shots must be >= 1')

        self.shots = shots  # TODO: Offer calculation based on metric
        self.n_neighbors = n_neighbors
        self.df = None
        self.items = None
        self.circle_coords_list = None

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

    def calculate_areas(self, df):
        df['p_area'] = 0.  # Area of the circle outside the polygon
        df['c_area'] = 0.  # Area of the circle intersected with other circles
        for i, row in df.iterrows():
            p_area, c_area = self.get_intersections(df, row['x'], row['y'], row['neighbors'], self.radius)

            df.loc[i, 'p_area'] = p_area
            df.loc[i, 'c_area'] = c_area

        df['wrong_area'] = df['p_area'] + df['c_area']
        return df

    @staticmethod
    def initialize_neighbors(selected_circles):
        df = pd.DataFrame(selected_circles, columns=['x', 'y'])

        df['neighbors'] = None
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
        step = self.radius / 2  # TODO: add this as an optional parameter
        for shot in range(self.shots):

            if shot > 0:
                df = self.calculate_neighbors(df)

            # TODO: Add circles not moving check (If many circles are not moving in consecutive iterations, break)
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
                if wrong_area_min < 1e-5:  # TODO: Add this as a parameter
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
        self.circle_coords_list = self.initialize_circles()
        selected_circles = self.select_circles(self.circle_coords_list)
        df = self.initialize_neighbors(selected_circles)
        df = self.calculate_areas(df)
        self.df = self.optimize(df)
        return self.df

    def score(self):
        """
        Included items area is used as penalty to the number of included items
        :return: computed score
        """
        if self.df is None:
            raise Exception('`pack` method must be called first')

        self.df['item'] = self.df.apply(lambda row: Point((row['x'], row['y'])).buffer(self.radius), axis=1)

        duplicate_area = 0
        for idx, row in self.df.iterrows():
            duplicate_area += row['item'].intersection(unary_union(self.df.drop(idx)['item'])).area
        duplicate_area /= 2

        included_items = unary_union(self.df['item'])
        intersected_area = included_items.intersection(self.polygon).area
        not_intersected_area = included_items.area - intersected_area
        wasted_area = not_intersected_area + duplicate_area

        return 1 + len(self.df) / len(self.circle_coords_list) + (wasted_area - intersected_area) / self.polygon.area


# TODO: Add Bee-Top initialization (Less waste of space)
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
            df = self.calculate_areas(df)
            df_wrong = df[df['wrong_area'] > 1e-10]  # TODO: Add this as a parameter
            if len(df_wrong) > 0:
                # TODO: Improve this computationally
                idx = df_wrong.sort_values('wrong_area', ascending=False).index[0]
                df.drop(idx, inplace=True)
                df['neighbors'] = df['neighbors'].apply(lambda l: [i for i in l if i != idx])
                # df = self.calculate_neighbors(df[['x', 'y']])
            else:
                break
        return df

    def score(self):
        if self.df is None:
            raise Exception('`pack` method must be called first')

        circles_area = len(self.df) * np.pi * self.radius ** 2
        return 1 - circles_area / self.polygon.area


class PolygonFillerBeeStrict(BeeInitialization, PolygonFillerStrict):
    pass


# noinspection PyMissingConstructor
class IrregularPacker:
    def __init__(self, container, items, intersection_threshold=60, max_iter=1000,
                 shots=1, n_neighbors=5, step=None, step_portion=2, queue_length=3,
                 n_rotations=12, init_threshold=90):
        self.container = container
        self.coords = np.array(self.container.exterior.coords)
        self.items = items
        self.intersection_threshold = intersection_threshold
        self.max_iter = max_iter  # TODO: Offer calculation based on polygon and circles size

        if shots < 1:
            raise ValueError('Number of shots must be >= 1')

        self.shots = shots  # TODO: Offer calculation based on metric
        self.n_neighbors = n_neighbors
        self.step = step
        if self.step is None:
            self.step = np.median(np.array([np.sqrt(item.area) for item in self.items])) / step_portion

        self.queue_length = queue_length
        self.rotations = np.linspace(0, 360, n_rotations + 1)[:n_rotations]
        self.init_threshold = init_threshold
        self.df = None

    # TODO: Add initialization from a grid of centroids spaced by a parameter from the set of shapes (median)
    # noinspection PyArgumentList
    def initialize_items(self):

        coords = self.coords.copy()
        while True:
            # Ensures to get the lowest-leftmost point of the polygon
            lowest_coords = coords[coords[:, 1] == coords[:, 1].min()]
            lowest_leftmost_coords = lowest_coords[lowest_coords[:, 0].argmin()]
            idx = np.argwhere(coords == lowest_leftmost_coords)[0, 0]
            item_coords = np.array(self.items[0].exterior.coords)
            transformed_coords = item_coords - item_coords.min(axis=0) + lowest_leftmost_coords
            item = shape({'type': 'Polygon', 'coordinates': [transformed_coords]})
            if item.centroid.intersects(self.container):
                break

            else:
                coords = np.delete(coords, idx, 0)

        # Get the leftmost-highest point of the polygon
        leftmost_coords = self.coords[self.coords[:, 0] == self.coords[:, 0].min()]
        leftmost_highest_coords = leftmost_coords[leftmost_coords[:, 1].argmax()]

        diff = leftmost_highest_coords - lowest_leftmost_coords
        if diff[0] == 0:
            m = 0
        else:
            m = diff[1] / diff[0]

        ref_coords = lowest_leftmost_coords.copy()
        initial_item = self.items[0]
        initial_item_coords = np.array(initial_item.exterior.coords)
        save_initial_item = True
        for idx, item in enumerate(self.items):
            item_coords = np.array(item.exterior.coords)
            transformed_coords = item_coords - item_coords.min(axis=0) + ref_coords

            if not save_initial_item:
                transformed_coords[:, 1] += initial_item_coords[:, 1].min() - transformed_coords[:, 1].min()

            item = shape({'type': 'Polygon', 'coordinates': [transformed_coords]})
            self.items[idx] = item

            if save_initial_item:
                initial_item = item
                initial_item_coords = np.array(initial_item.exterior.coords)
                save_initial_item = False

            if item.intersection(self.container).area / item.area * 100 >= self.init_threshold or item == initial_item:
                centroid = np.array(item.centroid.coords)
                max_y = np.array(item.exterior.coords)[:, 1].max()
                y_offset = max_y - centroid[0, 1]
                ref_coords = centroid - np.array([0, y_offset])

            else:
                centroid = np.array(initial_item.centroid.coords)
                max_y = np.array(initial_item.exterior.coords)[:, 1].max()
                min_x = np.array(initial_item.exterior.coords)[:, 0].min()
                y_offset = max_y - centroid[0, 1]
                x_offset = m * y_offset - (centroid[0, 0] - min_x)
                ref_coords = centroid + np.array([x_offset, y_offset])
                save_initial_item = True

    def select_items(self):
        selected_items = []
        for item in self.items:
            intersection = self.container.intersection(item).area / item.area * 100
            if intersection >= self.intersection_threshold:
                selected_items.append(item)

        return selected_items

    def calculate_neighbors(self, df):
        distances = pd.DataFrame(squareform(pdist(df.loc[:, ['y', 'x']])))
        distances.index = df.index
        distances.columns = df.index
        df['neighbors'] = None
        for idx, row in df.iterrows():
            df.at[idx, 'neighbors'] = distances.loc[idx].sort_values().head(self.n_neighbors + 1) \
                                          .index[1:].tolist()

        return df

    def initialize_neighbors(self, selected_items):
        selected_items_coords = np.array([item.centroid.coords for item in selected_items])[:, 0]
        df = pd.DataFrame(selected_items_coords, columns=['x', 'y'])
        df['item'] = selected_items
        return self.calculate_neighbors(df)

    @staticmethod
    def calculate_intersection(first_item, second_item):
        return first_item.intersection(second_item).area / second_item.area * 100

    def get_c_area(self, df, item, neighbors):
        c_area = 0
        for neighbor in neighbors:
            c_area += self.calculate_intersection(df.loc[neighbor, 'item'], item)

        return c_area

    def get_intersections(self, df, item, neighbors):
        p_area = 100 - self.calculate_intersection(self.container, item)
        c_area = self.get_c_area(df, item, neighbors)

        return p_area, c_area

    def calculate_areas(self, df):
        df['p_area'] = 0.  # Area of the circle outside the polygon
        df['c_area'] = 0.  # Area of the circle intersected with other circles
        for i, row in df.iterrows():
            p_area, c_area = self.get_intersections(df, row['item'], row['neighbors'])

            df.loc[i, 'p_area'] = p_area
            df.loc[i, 'c_area'] = c_area

        df['wrong_area'] = df['p_area'] + df['c_area']
        return df

    def optimize(self, df):
        for shot in range(self.shots):

            if shot > 0:
                df = self.calculate_neighbors(df)

            # TODO: Add circles not moving check (If many circles are not moving in consecutive iterations, break)
            df['locked'] = False
            queue = [-1] * self.queue_length
            for _ in range(self.max_iter):
                df_aux = df[~df['locked']].copy()
                if df_aux.empty:
                    break

                idx = df_aux['wrong_area'].idxmax()
                queue.append(idx)
                queue.pop(0)

                if queue == [idx] * self.queue_length:  # If the last n iterations have moved the same circle, lock it
                    df.loc[idx, 'locked'] = True

                row = df.loc[idx]

                y = row['y']
                x = row['x']
                up = 0, self.step
                right = self.step, 0
                down = 0, -self.step
                left = -self.step, 0

                wrong_area_min = row['wrong_area']
                if wrong_area_min < 1e-5:  # TODO: Add this as a parameter
                    break

                p_area_min = row['p_area']
                c_area_min = row['c_area']
                new_item = row['item']
                item_coords = np.array(new_item.exterior.coords)
                for direction in (up, right, down, left):
                    transformed_coords = item_coords + np.array(direction)
                    transformed_item = shape({'type': 'Polygon', 'coordinates': [transformed_coords]})
                    for rotation in self.rotations:
                        transformed_item = rotate(transformed_item, rotation)
                        p_area, c_area = self.get_intersections(df, transformed_item, row['neighbors'])
                        if p_area + c_area < wrong_area_min:
                            p_area_min = p_area
                            c_area_min = c_area
                            wrong_area_min = p_area + c_area
                            x = transformed_coords[0, 0]
                            y = transformed_coords[0, 1]
                            new_item = transformed_item

                df.loc[idx, 'x'] = x
                df.loc[idx, 'y'] = y
                df.loc[idx, 'item'] = new_item
                df.loc[idx, 'p_area'] = p_area_min
                df.loc[idx, 'c_area'] = c_area_min
                df.loc[idx, 'wrong_area'] = wrong_area_min

                for neighbor in row['neighbors']:
                    c_area = self.get_c_area(df, df.loc[neighbor, 'item'], df.loc[neighbor, 'neighbors'])
                    df.loc[neighbor, 'c_area'] = c_area
                    df.loc[neighbor, 'wrong_area'] = df.loc[neighbor, 'c_area'] + df.loc[neighbor, 'p_area']

        return df

    def pack(self):
        self.initialize_items()
        selected_items = self.select_items()
        df = self.initialize_neighbors(selected_items)
        df = self.calculate_areas(df)
        self.df = self.optimize(df)
        return self.df

    def score(self):
        """
        Included items area is used as penalty to the number of included items
        :return: computed score
        """
        if self.df is None:
            raise Exception('`pack` method must be called first')

        duplicate_area = 0
        for idx, row in self.df.iterrows():
            duplicate_area += row['item'].intersection(unary_union(self.df.drop(idx)['item'])).area
        duplicate_area /= 2

        included_items = unary_union(self.df['item'])
        intersected_area = included_items.intersection(self.container).area
        not_intersected_area = included_items.area - intersected_area
        not_included_items_area = sum([item.area for item in self.items if item not in self.df['item'].tolist()])
        wasted_area = not_included_items_area + not_intersected_area + duplicate_area

        return 1 + len(self.df) / len(self.items) + (wasted_area - intersected_area) / self.container.area


# TODO: Implement ordered initialization (bigger items first)
# TODO: Implement items addition to unused spaces
# TODO: Calculate max possible area
# TODO: Add random moves to the pieces after the first shot
class IrregularPackerStrict(IrregularPacker):
    """
    Variant to IrregularPacker in which overlapping items and items with parts outside
    the container are not allowed
    """
    def optimize(self, df):  # TODO: Add tqdm
        while True:
            df = super().optimize(df)
            df = self.calculate_neighbors(df)
            df = self.calculate_areas(df)
            df_wrong = df[df['wrong_area'] > 1e-10]  # TODO: Add this as a parameter
            if len(df_wrong) > 0:
                # TODO: Improve this computationally
                idx = df_wrong.sort_values('wrong_area', ascending=False).index[0]
                df.drop(idx, inplace=True)
                df['neighbors'] = df['neighbors'].apply(lambda l: [i for i in l if i != idx])
            else:
                break

        df['area'] = df['item'].apply(lambda item: item.area)
        df.sort_values('area', inplace=True)
        for idx, row in df.iterrows():
            if row['item'].intersection(unary_union(df.drop(idx)['item'])).area != 0.:
                df.drop(idx, inplace=True)

        return df

    def score(self):
        if self.df is None:
            raise Exception('`pack` method must be called first')

        not_included_area = sum([item.area for item in self.items if item not in self.df['item'].tolist()])

        return 1 + (not_included_area - self.df['area'].sum()) / self.container.area
