import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist
from shapely.geometry import shape, Point
from shapely.affinity import rotate, translate
from shapely.ops import unary_union
from tqdm.notebook import tqdm, trange


DISABLE_PROGRESS_BAR = bool(os.getenv('DISABLE_PROGRESS_BAR', False))


def get_coords(x, y):
    circle_coords_list = []
    for i in x:
        for j in y:
            circle_coords_list.append([i, j])

    return circle_coords_list


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


# noinspection PyMissingConstructor
class IrregularPacker:
    def __init__(self, container, items, intersection_threshold=60, max_iter=1000,
                 shots=1, n_neighbors=5, step=None, step_portion=2, queue_length=3,
                 n_rotations=12, init_threshold=90):
        self.container = container
        self.coords = np.array(self.container.exterior.coords)
        self.items = self.sort(items)
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
        self.rotations = np.linspace(-180, 180, n_rotations + 1)[:n_rotations]
        self.init_threshold = init_threshold
        self.df = None
        self.df_all = None

    @staticmethod
    def sort(items):
        return items

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
        items_coords = np.array([item.centroid.coords for item in self.items])[:, 0]
        self.df_all = pd.DataFrame(items_coords, columns=['x', 'y'])
        self.df_all['item'] = self.items
        df = self.df_all[self.df_all['item'].isin(selected_items)].copy()
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

    @staticmethod
    def drop_intersected(df):
        return df

    @staticmethod
    def drop_external(df):
        return df

    def optimize(self, df):
        for shot in range(self.shots):

            if shot > 0:
                df = self.calculate_neighbors(df)

            # TODO: Add circles not moving check (If many circles are not moving in consecutive iterations, break)
            df['r'] = 0.
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
                r = row['r']
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
                            r = rotation
                            new_item = transformed_item

                df.loc[idx, 'x'] = x
                df.loc[idx, 'y'] = y
                df.loc[idx, 'r'] = r
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


# TODO: Calculate max possible area
# TODO: Add random moves to the pieces after the first shot
class IrregularPackerStrict(IrregularPacker):
    """
    Variant to IrregularPacker in which overlapping items and items with parts outside
    the container are not allowed
    """

    @staticmethod
    def drop_intersected(df):
        df['area'] = df['item'].apply(lambda item: item.area)
        df.sort_values('area', inplace=True)
        for idx, row in df.iterrows():
            if row['item'].intersection(unary_union(df.drop(idx)['item'])).area != 0.:
                df.drop(idx, inplace=True)

        return df

    def drop_external(self, df):
        for idx, row in df.iterrows():
            if not row['item'].intersection(self.container).equals(row['item']):
                df.drop(idx, inplace=True)

        return df

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

        return self.drop_intersected(self.drop_external(df))

    def score(self):
        if self.df is None:
            raise Exception('`pack` method must be called first')

        return (self.container.area - self.df['area'].sum()) / self.container.area


class BiggerFirstInitialization:
    @staticmethod
    def sort(items):
        areas = [i.area for i in items]
        return [i for _, i in sorted(zip(areas, items), reverse=True)]


class IrregularPackerBF(BiggerFirstInitialization, IrregularPacker):
    pass


class IrregularPackerStrictBF(BiggerFirstInitialization, IrregularPackerStrict):
    pass


class GridInitialization:
    coords = None
    items = None
    container = None
    matching_coords = None
    spacing = None

    def initialize_items(self):
        x_max, y_max = self.coords.max(axis=0)
        x_min, y_min = self.coords.min(axis=0)

        # spacing = np.sqrt(np.median(np.array([i.area for i in self.items]))) / 2
        self.spacing = np.median(np.array([i.length for i in self.items])) / 4

        n_points_x = (x_max - x_min) / self.spacing
        n_points_y = (y_max - y_min) / self.spacing

        x_max = x_min + self.spacing * (n_points_x + 1)
        y_max = y_min + self.spacing * (n_points_y + 1)
        n_points_x = round((x_max - x_min) / self.spacing)
        n_points_y = round((y_max - y_min) / self.spacing)

        x = np.linspace(x_min, x_max, n_points_x + 1)
        y = np.linspace(y_min, y_max, n_points_y + 1)
        coords = get_coords(x, y)

        initial_coords = [
            coords[0][0] - 100 * self.spacing,
            coords[0][1] - 100 * self.spacing,
        ]

        self.matching_coords = []
        for coord in coords:
            if self.container.contains(Point(coord)):
                self.matching_coords.append(coord)

        self.items = [
            shape({
                'type': 'Polygon',
                'coordinates': [np.array(s.exterior.coords) - np.array(s.centroid.coords) + initial_coords]
            })
            for s in self.items
        ]

        for idx, centroid in enumerate(self.matching_coords):
            if idx == len(self.items):
                break

            item = self.items[idx]
            self.items[idx] = shape({
                'type': 'Polygon',
                'coordinates': [np.array(item.exterior.coords) - np.array(item.centroid.coords) + centroid]
            })


class IrregularPackerGridBF(BiggerFirstInitialization, GridInitialization, IrregularPacker):
    pass


class IrregularPackerStrictGridBF(BiggerFirstInitialization, GridInitialization, IrregularPackerStrict):
    pass


class Found(Exception):
    pass


# noinspection PyUnresolvedReferences,PyAttributeOutsideInit
class LocalSearch:
    def __init__(self, *args, n_search=1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_search = n_search

    def calculate_wrong_area(self, row):
        df_aux = pd.concat([self.df, row.to_frame().T])
        distances = np.linalg.norm((df_aux.loc[:, ['x', 'y']] - row[['x', 'y']]).astype(float), axis=1)
        neighbors = [i for _, i in sorted(zip(distances, df_aux.index))[1:self.n_neighbors + 1]]
        p_area, c_area = self.get_intersections(df_aux, row['item'], neighbors)
        return p_area + c_area, neighbors

    def optimize(self, df):
        df = super().optimize(df).copy()
        print("Finished first optimization step")
        df_out = self.df_all[~self.df_all.index.isin(df.index)].copy()

        df_optimum = df.copy()
        score = np.inf

        for _ in trange(self.n_search, disable=DISABLE_PROGRESS_BAR):
            self.df = df.copy()
            noise = np.random.uniform(-self.spacing, self.spacing, (len(df_out), 2))
            matching_coords = np.array(self.matching_coords)
            np.random.shuffle(matching_coords)
            matching_coords = np.resize(matching_coords, (len(df_out), 2))
            df_out.loc[:, ['x', 'y']] = noise + matching_coords

            df_out['item'] = df_out.apply(
                lambda r: translate(
                    r['item'],
                    r['x'] - r['item'].centroid.coords[0][0],
                    r['y'] - r['item'].centroid.coords[0][1]
                ),
                axis=1
            )

            df_out['rotation'] = np.random.uniform(0, 360, len(df_out))
            df_out['item'] = df_out.apply(lambda r: rotate(r['item'], r['rotation']), axis=1)

            df_out['wrong_area'] = df_out.apply(self.calculate_wrong_area, axis=1).str[0]

            up = 0, self.step
            right = self.step, 0
            down = 0, -self.step
            left = -self.step, 0
            for idx, row in df_out.sort_values('wrong_area').iterrows():
                y = row['y']
                x = row['x']
                new_item = row['item']
                wrong_area, neighbors = self.calculate_wrong_area(row)
                try:
                    if wrong_area < 1e-5:
                        raise Found

                    item_coords = np.array(new_item.exterior.coords)
                    for direction in (up, right, down, left):
                        transformed_coords = item_coords + np.array(direction)
                        transformed_item = shape({'type': 'Polygon', 'coordinates': [transformed_coords]})
                        for rotation in self.rotations:
                            transformed_item = rotate(transformed_item, rotation)
                            p_area, c_area = self.get_intersections(self.df, transformed_item, neighbors)
                            if p_area + c_area < 1e-5:
                                x = transformed_coords[0, 0]
                                y = transformed_coords[0, 1]
                                new_item = transformed_item
                                raise Found

                except Found:
                    self.df.loc[idx, 'x'] = x
                    self.df.loc[idx, 'y'] = y
                    self.df.loc[idx, 'item'] = new_item
                    self.df.loc[idx, 'p_area'] = 0
                    self.df.loc[idx, 'c_area'] = 0
                    self.df.loc[idx, 'wrong_area'] = 0

            new_score = self.score()
            if new_score < score:
                score = new_score
                df_optimum = self.df.copy()

        return self.drop_intersected(df_optimum)


class IrregularPackerGBFLS(LocalSearch, IrregularPackerGridBF):
    @staticmethod
    def drop_intersected(df):
        return df


class IrregularPackerStrictGBFLS(LocalSearch, IrregularPackerStrictGridBF):
    pass


# noinspection PyMissingConstructor
class CirclePacker(IrregularPacker):
    """
    This class takes polygon coordinates and returns coordinates of circles
    filling the given polygon
    """

    def __init__(self, container, radius, intersection_threshold=60, max_iter=1000,
                 shots=1, n_neighbors=5, step_portion=2, queue_length=3):
        """
        Args:
            container: Shapely geometry
            radius: Circle radius in kilometers
            intersection_threshold: Exclude initial circles when their percentage of intersection with the polygon is\
             less than threshold
            max_iter: Max number of circle movement iterations
            shots: Number of algorithm runs (Useful for refining)
            n_neighbors: Number of neighbors to consider in case shots > 1
        """

        self.container = container
        self.coords = np.array(self.container.exterior.coords)
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
        self.initial_length = None

        self.step = self.radius / step_portion

        self.queue_length = queue_length
        self.rotations = [0]
        self.df = None
        self.df_all = None

    @staticmethod
    def get_circle_coords(x, y):
        return get_coords(x, y)

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
            intersection = self.calculate_intersection(
                self.container,
                Point((circle_coords[0], circle_coords[1])).buffer(self.radius)
            )
            if intersection >= self.intersection_threshold:
                intersections.append(intersection)
                selected_circles.append(circle_coords)

        return selected_circles

    def initialize_neighbors(self, selected_circles):
        df = pd.DataFrame(selected_circles, columns=['x', 'y'])
        df['item'] = df.apply(lambda r: Point((r['x'], r['y'])).buffer(self.radius), axis=1)
        self.df_all = df.copy()
        self.df_all.index += len(self.df_all)
        return self.calculate_neighbors(df)

    def pack(self):
        self.circle_coords_list = self.initialize_circles()
        self.initial_length = len(self.circle_coords_list)
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

        self.df['item'] = self.df.apply(lambda r: Point((r['x'], r['y'])).buffer(self.radius), axis=1)

        duplicate_area = 0
        for idx, row in self.df.iterrows():
            duplicate_area += row['item'].intersection(unary_union(self.df.drop(idx)['item'])).area
        duplicate_area /= 2

        included_items = unary_union(self.df['item'])
        intersected_area = included_items.intersection(self.container).area
        not_intersected_area = included_items.area - intersected_area
        wasted_area = not_intersected_area + duplicate_area

        return 1 + len(self.df) / self.initial_length + (wasted_area - intersected_area) / self.container.area


class CirclePackerStrict(CirclePacker, IrregularPackerStrict):
    def score(self):
        if self.df is None:
            raise Exception('`pack` method must be called first')

        circles_area = len(self.df) * np.pi * self.radius ** 2
        return 1 - circles_area / self.container.area


class CirclePackerBee(BeeInitialization, CirclePacker):
    pass


class CirclePackerBeeStrict(BeeInitialization, CirclePackerStrict):
    pass


# noinspection PyUnresolvedReferences
class CircleLocalSearch(LocalSearch):
    matching_coords = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spacing = self.radius

    def optimize(self, df):
        self.matching_coords = df[['x', 'y']].values
        return super().optimize(df)

    @staticmethod
    def drop_intersected(df):
        return df


class CirclePackerBeeLS(CircleLocalSearch, CirclePackerBee):
    pass


class CirclePackerBeeStrictLS(CircleLocalSearch, CirclePackerBeeStrict):
    pass


class IrregularPackerPSO:
    min_radius = -180
    max_radius = 180
    global_optimum = np.inf
    global_optimum_position = None
    packer = IrregularPacker

    def __init__(self, container, items, *args, phi_p=1, phi_g=1, w=1, n_particles=10, n_iterations=10):
        self.container = container
        self.items = items
        self.phi_p = phi_p
        self.phi_g = phi_g
        self.w = w
        self.n_particles = n_particles
        self.n_iterations = n_iterations

    def get_packer(self):
        return self.packer(self.container, self.items)

    def get_items(self):
        return self.items

    # noinspection PyArgumentList
    def get_random_uniform(self):
        return np.random.uniform(
            np.array(self.container.exterior.coords).min(axis=0).tolist() + [self.min_radius],
            np.array(self.container.exterior.coords).max(axis=0).tolist() + [self.max_radius],
            (len(self.items), 3)
        )

    def optimize(self, _, score, particle_optimum, particle_position, particle_optimum_position, i):
        particle_optimum[i] = score
        particle_optimum_position[i] = particle_position[i].copy()

        if score < self.global_optimum:
            self.global_optimum = score
            self.global_optimum_position = particle_position[i].copy()

    # noinspection PyUnresolvedReferences
    def pack(self):
        particle_optimum = []
        particle_optimum_position = []
        particle_position = []
        particle_velocity = []
        packer = self.get_packer()
        is_strict = hasattr(packer, 'drop_external') and hasattr(packer, 'drop_intersected')

        for _ in tqdm(range(self.n_particles), disable=DISABLE_PROGRESS_BAR):
            self.items = self.get_items()
            initial_position = self.get_random_uniform()
            df = pd.DataFrame(initial_position, columns=['x', 'y', 'r'])
            df['item'] = self.items

            df['item'] = df.apply(
                lambda r: rotate(
                    translate(
                        r['item'],
                        r['x'] - r['item'].centroid.coords[0][0],
                        r['y'] - r['item'].centroid.coords[0][1]
                    ),
                    r['r']
                ),
                axis=1
            )

            particle_position.append(df.copy())
            particle_optimum_position.append(df.copy())

            if is_strict:
                df_feasible = packer.drop_external(df.copy())
                packer.drop_intersected(df_feasible)
                packer.df = df_feasible
            else:
                packer.df = df.copy()

            if hasattr(packer, 'initial_length'):
                packer.initial_length = self.initial_length

            score = packer.score()
            particle_optimum.append(score)
            particle_velocity.append(self.get_random_uniform())

            if score < self.global_optimum:
                self.global_optimum = score
                self.global_optimum_position = df.copy()

        # TODO: Add crossing for solutions with different lengths
        # particle_velocity[i] = (
        #         self.w * particle_velocity[i] +
        #         self.phi_p * rp * (
        #                 particle_optimum_position[i][['x', 'y', 'r']].iloc[:len(particle_position[i])] -
        #                 particle_position[i][['x', 'y', 'r']].iloc[:len(particle_optimum_position[i])]
        #         ).values +
        #
        #         self.phi_g * rg * (
        #                 self.global_optimum_position[['x', 'y', 'r']].iloc[:len(particle_position[i])] -
        #                 particle_position[i][['x', 'y', 'r']].iloc[:len(self.global_optimum_position[i])]
        #         ).values
        # )
        # particle_velocity[i][:, 2] %= self.max_radius
        # if len(particle_velocity[i]) > len(particle_position[i]):
        # particle_position[i][['x', 'y', 'r']] += particle_velocity[i]

        for _ in tqdm(range(self.n_iterations), disable=DISABLE_PROGRESS_BAR):
            for i in range(self.n_particles):
                rp = np.random.uniform(0, 1, (len(self.items), 3))
                rg = np.random.uniform(0, 1, (len(self.items), 3))
                particle_velocity[i] = (
                        self.w * particle_velocity[i] +
                        self.phi_p * rp * (
                                particle_optimum_position[i][['x', 'y', 'r']] -
                                particle_position[i][['x', 'y', 'r']]
                        ).values +

                        self.phi_g * rg * (
                                self.global_optimum_position[['x', 'y', 'r']] -
                                particle_position[i][['x', 'y', 'r']]
                        ).values
                )

                particle_position[i]['r_old'] = particle_position[i]['r']
                particle_position[i][['x', 'y', 'r']] += particle_velocity[i]

                particle_position[i]['r'] = particle_position[i]['r'].apply(
                    lambda x: x - 360 if x > 180 else x
                ).apply(
                    lambda x: x + 360 if x < -180 else x
                )

                particle_position[i]['item'] = particle_position[i].apply(
                    lambda r: rotate(
                        translate(
                            r['item'],
                            r['x'] - r['item'].centroid.coords[0][0],
                            r['y'] - r['item'].centroid.coords[0][1]
                        ),
                        r['r'] - r['r_old']
                    ),
                    axis=1
                )

                if is_strict:
                    df_feasible = packer.drop_external(particle_position[i].copy())
                    packer.drop_intersected(df_feasible)
                    packer.df = df_feasible
                else:
                    packer.df = particle_position[i].copy()

                score = packer.score()
                if score < particle_optimum[i]:
                    self.optimize(packer, score, particle_optimum, particle_position, particle_optimum_position, i)
                    # particle_optimum[i] = score
                    # particle_optimum_position[i] = particle_position[i].copy()
                    #
                    # if score < self.global_optimum:
                    #     self.global_optimum = score
                    #     self.global_optimum_position = particle_position[i].copy()


class LocalSearchPSO:
    particle_packer = None
    container = None
    global_optimum = np.inf
    global_optimum_position = None
    get_particle_packer = None

    def __init__(self, *args, get_particle_packer=None, **kwargs):
        super().__init__(*args, **kwargs)

        if get_particle_packer is not None:
            self.get_particle_packer = get_particle_packer

        if self.get_particle_packer is None:
            self.get_particle_packer = lambda c, i: self.particle_packer(c, i)

    def optimize(self, packer, score, particle_optimum, particle_position, particle_optimum_position, i):
        items = particle_position[i]['item'].tolist()
        if hasattr(self, 'radius'):
            items = [[item.centroid.coords[0][0], item.centroid.coords[0][1]] for item in items]
        particle_packer = self.get_particle_packer(self.container, items)
        df = particle_packer.initialize_neighbors(items)
        df = particle_packer.calculate_areas(df)
        df = particle_packer.optimize(df)
        df_feasible = packer.drop_external(df.copy())
        packer.drop_intersected(df_feasible)
        packer.df = df_feasible.copy()

        new_score = packer.score()

        df['r'] = (df['r'] + particle_position[i]['r']).apply(
            lambda x: x - 360 if x > 180 else x
        ).apply(
            lambda x: x + 360 if x < -180 else x
        )

        if new_score < score:
            score = new_score
            particle_optimum[i] = score
            particle_optimum_position[i] = df[['x', 'y', 'r', 'item']].copy()

        if score < self.global_optimum:
            self.global_optimum = score
            self.global_optimum_position = df[['x', 'y', 'r', 'item']].copy()


class IrregularPackerPSOLS(LocalSearchPSO, IrregularPackerPSO):
    particle_packer = IrregularPacker


class IrregularPackerStrictPSO(IrregularPackerPSO):
    packer = IrregularPackerStrict


class IrregularPackerStrictPSOLS(IrregularPackerPSOLS):
    packer = IrregularPackerStrict
    particle_packer = IrregularPacker


class CirclePackerPSO(IrregularPackerPSO):
    min_radius = 0.
    max_radius = 0.
    global_optimum = np.inf
    packer = CirclePackerBee

    def __init__(self, container, radius, n_circle_iter=3, **kwargs):
        self.n_circles = None
        self.n_circle_iter = n_circle_iter
        self.radius = radius

        packer = CirclePackerBeeStrict(container, self.radius)
        self.lower_bound = len(packer.pack()) - 2
        if self.lower_bound <= 0:
            self.lower_bound = 1
        self.initial_length = packer.initial_length

        packer = CirclePackerBee(container, self.radius, intersection_threshold=20, shots=5)
        self.upper_bound = len(packer.select_circles(packer.initialize_circles())) + 2

        items = None
        super().__init__(container, items, **kwargs)

    def get_packer(self):
        return self.packer(self.container, self.radius)

    def get_items(self):
        # n_circles = np.random.randint(self.lower_bound, self.upper_bound, 1)[0]
        return [Point((0, 0)).buffer(self.radius) for _ in range(self.n_circles)]

    def pack(self):
        best_score = self.global_optimum
        best_position = None

        for n_circles in set(np.linspace(self.lower_bound, self.upper_bound, self.n_circle_iter).astype(int)):
            self.n_circles = n_circles
            self.global_optimum = np.inf
            super().pack()
            if self.global_optimum < best_score:
                best_score = self.global_optimum
                best_position = self.global_optimum_position.copy()

        self.global_optimum = best_score
        self.global_optimum_position = best_position


class CirclePackerStrictPSO(CirclePackerPSO):
    packer = CirclePackerBeeStrict


class CirclePackerPSOLS(LocalSearchPSO, CirclePackerPSO):
    particle_packer = CirclePacker

    def get_particle_packer(self, c, _):
        return self.particle_packer(c, self.radius)


class CirclePackerStrictPSOLS(CirclePackerPSOLS):
    packer = CirclePackerBeeStrict
