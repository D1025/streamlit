import math
import streamlit as streamlit
import pandas as pandas
import pydeck as pydeck
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium


class CentroidCalculator:
    def __init__(self, points=None):
        self._pts = []
        if points:
            self.extend(points)
    def add(self, x, y, transport_rate, mass):
        self._pts.append((float(x), float(y), float(transport_rate), float(mass)))
    def extend(self, points):
        for p in points:
            self._pts.append((float(p[0]), float(p[1]), float(p[2]), float(p[3])))
    def centroid(self):
        points_list = list(self._pts)
        if not points_list:
            return 0.0, 0.0
        total_weight = 0.0
        weighted_x_sum = 0.0
        weighted_y_sum = 0.0
        for x_coordinate, y_coordinate, transport_rate, mass in points_list:
            point_weight = float(transport_rate) * float(mass)
            total_weight += point_weight
            weighted_x_sum += point_weight * float(x_coordinate)
            weighted_y_sum += point_weight * float(y_coordinate)
        if abs(total_weight) < 1e-12:
            points_count = len(points_list)
            if points_count == 0:
                return 0.0, 0.0
            average_x = sum(p[0] for p in points_list) / points_count
            average_y = sum(p[1] for p in points_list) / points_count
            return float(average_x), float(average_y)
        centroid_x = weighted_x_sum / total_weight
        centroid_y = weighted_y_sum / total_weight
        return float(centroid_x), float(centroid_y)
    def weighted_euclidean_distance_sum(self, centroid_x, centroid_y):
        points_list = list(self._pts)
        if not points_list:
            return 0.0
        weighted_distance_sum = 0.0
        for x_coordinate, y_coordinate, transport_rate, mass in points_list:
            point_weight = float(transport_rate) * float(mass)
            euclidean_distance = math.sqrt((float(centroid_x) - float(x_coordinate)) ** 2 + (float(centroid_y) - float(y_coordinate)) ** 2)
            weighted_distance_sum += point_weight * euclidean_distance
        return float(weighted_distance_sum)


def normalize_column_names(input_dataframe):
    normalized_dataframe = input_dataframe.copy()
    normalized_dataframe.columns = [str(column).strip().lower() for column in normalized_dataframe.columns]
    return normalized_dataframe


def choose_existing_column(normalized_dataframe, candidate_names):
    for candidate_name in candidate_names:
        if candidate_name in normalized_dataframe.columns:
            return candidate_name
    return None


def ensure_points_dataframe(points_dataframe):
    if points_dataframe is None:
        return pandas.DataFrame(columns=["longitude", "latitude", "transport_rate", "mass"])
    normalized_dataframe = normalize_column_names(points_dataframe)

    longitude_column = choose_existing_column(normalized_dataframe, ["longitude", "lon", "x"])
    latitude_column = choose_existing_column(normalized_dataframe, ["latitude", "lat", "y"])
    transport_rate_column = choose_existing_column(normalized_dataframe, ["transport_rate", "transport", "rate", "stawka_transportowa", "stawka", "st"])
    mass_column = choose_existing_column(normalized_dataframe, ["mass", "masa", "m"])

    if longitude_column is None or latitude_column is None:
        if len(normalized_dataframe.columns) >= 2:
            inferred_longitude_column = normalized_dataframe.columns[0]
            inferred_latitude_column = normalized_dataframe.columns[1]
            longitude_column = longitude_column or inferred_longitude_column
            latitude_column = latitude_column or inferred_latitude_column

    if transport_rate_column is None:
        normalized_dataframe["transport_rate"] = 1.0
        transport_rate_column = "transport_rate"

    if mass_column is None:
        normalized_dataframe["mass"] = 1.0
        mass_column = "mass"

    if longitude_column is None:
        normalized_dataframe["longitude"] = []
        longitude_column = "longitude"
    if latitude_column is None:
        normalized_dataframe["latitude"] = []
        latitude_column = "latitude"

    selected_dataframe = normalized_dataframe[[longitude_column, latitude_column, transport_rate_column, mass_column]].copy()
    selected_dataframe.columns = ["longitude", "latitude", "transport_rate", "mass"]

    selected_dataframe["longitude"] = pandas.to_numeric(selected_dataframe["longitude"], errors="coerce")
    selected_dataframe["latitude"] = pandas.to_numeric(selected_dataframe["latitude"], errors="coerce")
    selected_dataframe["transport_rate"] = pandas.to_numeric(selected_dataframe["transport_rate"], errors="coerce")
    selected_dataframe["mass"] = pandas.to_numeric(selected_dataframe["mass"], errors="coerce")

    selected_dataframe = selected_dataframe.dropna(subset=["longitude", "latitude"]).reset_index(drop=True)
    selected_dataframe["transport_rate"] = selected_dataframe["transport_rate"].fillna(1.0)
    selected_dataframe["mass"] = selected_dataframe["mass"].fillna(1.0)

    return selected_dataframe


def append_point(points_dataframe, longitude, latitude, transport_rate, mass):
    updated_points_dataframe = ensure_points_dataframe(points_dataframe)
    new_row_dataframe = pandas.DataFrame([{"longitude": float(longitude), "latitude": float(latitude), "transport_rate": float(transport_rate), "mass": float(mass)}])
    updated_points_dataframe = pandas.concat([updated_points_dataframe, new_row_dataframe], ignore_index=True)
    return updated_points_dataframe


def read_points_from_uploaded_file(uploaded_file):
    if uploaded_file is None:
        return pandas.DataFrame(columns=["longitude", "latitude", "transport_rate", "mass"])
    uploaded_name = str(uploaded_file.name).lower()
    try:
        if uploaded_name.endswith(".csv"):
            uploaded_dataframe = pandas.read_csv(uploaded_file)
        elif uploaded_name.endswith(".xlsx") or uploaded_name.endswith(".xls"):
            uploaded_dataframe = pandas.read_excel(uploaded_file)
        else:
            uploaded_dataframe = pandas.read_csv(uploaded_file)
    except Exception:
        return pandas.DataFrame(columns=["longitude", "latitude", "transport_rate", "mass"])
    return ensure_points_dataframe(uploaded_dataframe)


def compute_center_of_gravity(points_dataframe):
    points_dataframe = ensure_points_dataframe(points_dataframe)
    centroid_calculator = CentroidCalculator()
    for _, row in points_dataframe.iterrows():
        centroid_calculator.add(row["longitude"], row["latitude"], row["transport_rate"], row["mass"])
    centroid_longitude, centroid_latitude = centroid_calculator.centroid()
    weighted_distance_sum = centroid_calculator.weighted_euclidean_distance_sum(centroid_longitude, centroid_latitude)
    return centroid_longitude, centroid_latitude, weighted_distance_sum


def get_map_center(points_dataframe, centroid_longitude, centroid_latitude):
    points_dataframe = ensure_points_dataframe(points_dataframe)
    if len(points_dataframe) > 0:
        last_point_longitude = float(points_dataframe.iloc[-1]["longitude"])
        last_point_latitude = float(points_dataframe.iloc[-1]["latitude"])
        return last_point_longitude, last_point_latitude
    if abs(float(centroid_longitude)) > 1e-9 or abs(float(centroid_latitude)) > 1e-9:
        return float(centroid_longitude), float(centroid_latitude)
    return 21.0122, 52.2297


def build_polyline_path(points_dataframe):
    points_dataframe = ensure_points_dataframe(points_dataframe)
    polyline_path = []
    for _, row in points_dataframe.iterrows():
        polyline_path.append([float(row["longitude"]), float(row["latitude"])])
    if len(polyline_path) >= 3 and polyline_path[0] != polyline_path[-1]:
        polyline_path.append(polyline_path[0])
    return polyline_path


def compute_point_distances(points_dataframe, centroid_longitude, centroid_latitude):
    points_dataframe = ensure_points_dataframe(points_dataframe)
    distances_dataframe = points_dataframe.copy()
    distances = []
    weighted_distances = []
    for _, row in distances_dataframe.iterrows():
        euclidean_distance = math.sqrt((float(centroid_longitude) - float(row["longitude"])) ** 2 + (float(centroid_latitude) - float(row["latitude"])) ** 2)
        point_weight = float(row["transport_rate"]) * float(row["mass"])
        distances.append(float(euclidean_distance))
        weighted_distances.append(float(point_weight) * float(euclidean_distance))
    distances_dataframe["euclidean_distance"] = distances
    distances_dataframe["weighted_euclidean_distance"] = weighted_distances
    return distances_dataframe


def extract_marker_positions_from_drawings(all_drawings):
    marker_positions = []
    if all_drawings is None:
        return marker_positions
    if isinstance(all_drawings, dict):
        drawings_iterable = list(all_drawings.values())
    elif isinstance(all_drawings, list):
        drawings_iterable = list(all_drawings)
    else:
        return marker_positions
    for drawing_feature in drawings_iterable:
        if not isinstance(drawing_feature, dict):
            continue
        geometry = drawing_feature.get("geometry")
        if not isinstance(geometry, dict):
            continue
        if geometry.get("type") != "Point":
            continue
        coordinates = geometry.get("coordinates")
        if not isinstance(coordinates, (list, tuple)) or len(coordinates) < 2:
            continue
        longitude_value = coordinates[0]
        latitude_value = coordinates[1]
        try:
            marker_positions.append((float(longitude_value), float(latitude_value)))
        except Exception:
            continue
    return marker_positions


def synchronize_points_dataframe_with_marker_positions(points_dataframe, marker_positions, default_transport_rate, default_mass):
    points_dataframe = ensure_points_dataframe(points_dataframe)
    marker_positions = list(marker_positions or [])
    if len(marker_positions) == 0:
        return pandas.DataFrame(columns=["longitude", "latitude", "transport_rate", "mass"])
    if len(points_dataframe) == 0:
        created_rows = []
        for longitude_value, latitude_value in marker_positions:
            created_rows.append({"longitude": float(longitude_value), "latitude": float(latitude_value), "transport_rate": float(default_transport_rate), "mass": float(default_mass)})
        return ensure_points_dataframe(pandas.DataFrame(created_rows))
    existing_positions = [(float(row["longitude"]), float(row["latitude"])) for _, row in points_dataframe.iterrows()]
    existing_rows = [{"longitude": float(row["longitude"]), "latitude": float(row["latitude"]), "transport_rate": float(row["transport_rate"]), "mass": float(row["mass"])} for _, row in points_dataframe.iterrows()]
    all_pairs = []
    for existing_index, (existing_longitude, existing_latitude) in enumerate(existing_positions):
        for marker_index, (marker_longitude, marker_latitude) in enumerate(marker_positions):
            distance_value = math.sqrt((existing_longitude - marker_longitude) ** 2 + (existing_latitude - marker_latitude) ** 2)
            all_pairs.append((distance_value, existing_index, marker_index))
    all_pairs.sort(key=lambda item: float(item[0]))
    matched_existing_indices = set()
    matched_marker_indices = set()
    matched_existing_to_marker = {}
    for _, existing_index, marker_index in all_pairs:
        if existing_index in matched_existing_indices:
            continue
        if marker_index in matched_marker_indices:
            continue
        matched_existing_indices.add(existing_index)
        matched_marker_indices.add(marker_index)
        matched_existing_to_marker[existing_index] = marker_index
        if len(matched_existing_indices) == len(existing_positions) or len(matched_marker_indices) == len(marker_positions):
            break
    updated_rows = []
    for existing_index in range(len(existing_rows)):
        if existing_index not in matched_existing_to_marker:
            continue
        marker_index = matched_existing_to_marker[existing_index]
        marker_longitude, marker_latitude = marker_positions[marker_index]
        updated_rows.append({"longitude": float(marker_longitude), "latitude": float(marker_latitude), "transport_rate": float(existing_rows[existing_index]["transport_rate"]), "mass": float(existing_rows[existing_index]["mass"])})
    for marker_index in range(len(marker_positions)):
        if marker_index in matched_marker_indices:
            continue
        marker_longitude, marker_latitude = marker_positions[marker_index]
        updated_rows.append({"longitude": float(marker_longitude), "latitude": float(marker_latitude), "transport_rate": float(default_transport_rate), "mass": float(default_mass)})
    return ensure_points_dataframe(pandas.DataFrame(updated_rows))


def points_dataframe_signature(points_dataframe):
    points_dataframe = ensure_points_dataframe(points_dataframe)
    signature_items = []
    for _, row in points_dataframe.iterrows():
        signature_items.append((round(float(row["longitude"]), 8), round(float(row["latitude"]), 8), round(float(row["transport_rate"]), 8), round(float(row["mass"]), 8)))
    return tuple(signature_items)


streamlit.set_page_config(page_title="Środek ciężkości logistyczny", layout="wide")

if "points_dataframe" not in streamlit.session_state:
    streamlit.session_state["points_dataframe"] = pandas.DataFrame(columns=["longitude", "latitude", "transport_rate", "mass"])

if "map_default_transport_rate" not in streamlit.session_state:
    streamlit.session_state["map_default_transport_rate"] = 1.0

if "map_default_mass" not in streamlit.session_state:
    streamlit.session_state["map_default_mass"] = 1.0

if "map_center_latitude" not in streamlit.session_state:
    streamlit.session_state["map_center_latitude"] = 52.2297

if "map_center_longitude" not in streamlit.session_state:
    streamlit.session_state["map_center_longitude"] = 21.0122

if "map_zoom_level" not in streamlit.session_state:
    streamlit.session_state["map_zoom_level"] = 11

if "map_marker_positions_snapshot" not in streamlit.session_state:
    streamlit.session_state["map_marker_positions_snapshot"] = ()

streamlit.title("Lokalizacja punktu logistycznego metodą środka ciężkości")
streamlit.write("Współrzędne wyznaczane są wzorami: X = Σ(ST·M·X) / Σ(ST·M) oraz Y = Σ(ST·M·Y) / Σ(ST·M). Odległości liczone są metryką euklidesową: d = √((X−Xn)² + (Y−Yn)²).")

control_column, map_column = streamlit.columns([1, 2], gap="large")

with control_column:
    streamlit.subheader("Dane wejściowe")
    uploaded_file = streamlit.file_uploader("Wczytaj punkty z pliku (CSV/XLSX). Kolumny: longitude/latitude oraz opcjonalnie transport_rate/mass", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None:
        uploaded_points_dataframe = read_points_from_uploaded_file(uploaded_file)
        if len(uploaded_points_dataframe) > 0:
            streamlit.session_state["points_dataframe"] = uploaded_points_dataframe

    points_dataframe = ensure_points_dataframe(streamlit.session_state["points_dataframe"])

    data_editor_function = getattr(streamlit, "data_editor", None)
    if data_editor_function is not None:
        edited_points_dataframe = data_editor_function(
            points_dataframe,
            num_rows="dynamic",
            use_container_width=True
        )
        streamlit.session_state["points_dataframe"] = ensure_points_dataframe(edited_points_dataframe)
        points_dataframe = ensure_points_dataframe(streamlit.session_state["points_dataframe"])
    else:
        streamlit.dataframe(points_dataframe, use_container_width=True)

    streamlit.divider()
    streamlit.subheader("Dodaj punkt ręcznie")
    manual_longitude = streamlit.number_input("X (longitude)", value=21.012200, format="%.6f")
    manual_latitude = streamlit.number_input("Y (latitude)", value=52.229700, format="%.6f")
    manual_transport_rate = streamlit.number_input("ST (stawka transportowa)", value=1.000000, min_value=0.0, format="%.6f")
    manual_mass = streamlit.number_input("M (masa)", value=1.000000, min_value=0.0, format="%.6f")
    add_manual_point_button = streamlit.button("Dodaj punkt", use_container_width=True)

    if add_manual_point_button:
        streamlit.session_state["points_dataframe"] = append_point(streamlit.session_state["points_dataframe"], manual_longitude, manual_latitude, manual_transport_rate, manual_mass)

    clear_points_button = streamlit.button("Wyczyść wszystkie punkty", use_container_width=True)
    if clear_points_button:
        streamlit.session_state["points_dataframe"] = pandas.DataFrame(columns=["longitude", "latitude", "transport_rate", "mass"])
        streamlit.session_state["map_marker_positions_snapshot"] = ()

    points_dataframe = ensure_points_dataframe(streamlit.session_state["points_dataframe"])
    centroid_longitude, centroid_latitude, weighted_distance_sum = compute_center_of_gravity(points_dataframe)

    streamlit.divider()
    streamlit.subheader("Wynik")
    if len(points_dataframe) == 0:
        streamlit.info("Dodaj co najmniej jeden punkt, aby wyliczyć X i Y.")
    else:
        streamlit.metric("X (longitude)", f"{centroid_longitude:.6f}")
        streamlit.metric("Y (latitude)", f"{centroid_latitude:.6f}")
        streamlit.metric("Suma ważona odległości euklidesowych", f"{weighted_distance_sum:.6f}")

        distances_dataframe = compute_point_distances(points_dataframe, centroid_longitude, centroid_latitude)
        streamlit.subheader("Odległości (metryka euklidesowa)")
        streamlit.dataframe(distances_dataframe, use_container_width=True)

    streamlit.divider()
    streamlit.subheader("Domyślne wartości dla punktów dodanych z mapy")
    streamlit.session_state["map_default_transport_rate"] = streamlit.number_input("Domyślne ST", value=float(streamlit.session_state["map_default_transport_rate"]), min_value=0.0, format="%.6f")
    streamlit.session_state["map_default_mass"] = streamlit.number_input("Domyślne M", value=float(streamlit.session_state["map_default_mass"]), min_value=0.0, format="%.6f")
    streamlit.write("Dodawanie: wybierz narzędzie markera na mapie i kliknij w miejsce punktu. Przesuwanie/usuwanie: użyj trybu edycji/usuwania w panelu narzędzi mapy.")

with map_column:
    points_dataframe = ensure_points_dataframe(streamlit.session_state["points_dataframe"])
    centroid_longitude, centroid_latitude, weighted_distance_sum = compute_center_of_gravity(points_dataframe)
    map_center_longitude, map_center_latitude = get_map_center(points_dataframe, centroid_longitude, centroid_latitude)
    polyline_path = build_polyline_path(points_dataframe)

    streamlit.subheader("Mapa")
    if "map_center_latitude" not in streamlit.session_state or "map_center_longitude" not in streamlit.session_state:
        streamlit.session_state["map_center_latitude"] = float(map_center_latitude)
        streamlit.session_state["map_center_longitude"] = float(map_center_longitude)
    if "map_zoom_level" not in streamlit.session_state:
        streamlit.session_state["map_zoom_level"] = 11

    folium_map = folium.Map(
        location=[float(streamlit.session_state["map_center_latitude"]), float(streamlit.session_state["map_center_longitude"])],
        zoom_start=int(streamlit.session_state["map_zoom_level"]),
        control_scale=True
    )

    editable_feature_group = folium.FeatureGroup(name="Punkty")
    editable_feature_group.add_to(folium_map)

    for _, row in points_dataframe.iterrows():
        folium.Marker(
            location=[float(row["latitude"]), float(row["longitude"])],
            tooltip=f"Y={float(row['latitude']):.6f}, X={float(row['longitude']):.6f}"
        ).add_to(editable_feature_group)

    Draw(
        export=False,
        feature_group=editable_feature_group,
        draw_options={
            "polyline": False,
            "polygon": False,
            "rectangle": False,
            "circle": False,
            "circlemarker": False
        },
        edit_options={}
    ).add_to(folium_map)

    if len(polyline_path) >= 4:
        folium.PolyLine(
            locations=[[latitude, longitude] for longitude, latitude in polyline_path],
            weight=3
        ).add_to(folium_map)

    if len(points_dataframe) > 0:
        folium.CircleMarker(
            location=[float(centroid_latitude), float(centroid_longitude)],
            radius=14,
            color="red",
            fill=True,
            fill_color="red",
            fill_opacity=0.95
        ).add_to(folium_map)

    map_interaction = st_folium(
        folium_map,
        height=600,
        use_container_width=True,
        key="interactive_folium_map",
        returned_objects=["all_drawings", "last_active_drawing", "center", "zoom"],
        center=[float(streamlit.session_state["map_center_latitude"]), float(streamlit.session_state["map_center_longitude"])],
        zoom=int(streamlit.session_state["map_zoom_level"])
    )

    if isinstance(map_interaction, dict):
        returned_center = map_interaction.get("center")
        returned_zoom = map_interaction.get("zoom")
        if isinstance(returned_center, dict) and "lat" in returned_center and "lng" in returned_center:
            try:
                streamlit.session_state["map_center_latitude"] = float(returned_center["lat"])
                streamlit.session_state["map_center_longitude"] = float(returned_center["lng"])
            except Exception:
                pass
        if returned_zoom is not None:
            try:
                streamlit.session_state["map_zoom_level"] = int(returned_zoom)
            except Exception:
                pass

        all_drawings = map_interaction.get("all_drawings")
        marker_positions = extract_marker_positions_from_drawings(all_drawings)
        marker_positions_signature = tuple((round(float(longitude_value), 8), round(float(latitude_value), 8)) for longitude_value, latitude_value in marker_positions)

        if marker_positions_signature != tuple(streamlit.session_state.get("map_marker_positions_snapshot", ())):
            previous_points_dataframe = ensure_points_dataframe(streamlit.session_state["points_dataframe"])
            previous_signature = points_dataframe_signature(previous_points_dataframe)
            synchronized_points_dataframe = synchronize_points_dataframe_with_marker_positions(
                previous_points_dataframe,
                marker_positions,
                float(streamlit.session_state["map_default_transport_rate"]),
                float(streamlit.session_state["map_default_mass"])
            )
            new_signature = points_dataframe_signature(synchronized_points_dataframe)
            streamlit.session_state["map_marker_positions_snapshot"] = marker_positions_signature
            if new_signature != previous_signature:
                streamlit.session_state["points_dataframe"] = synchronized_points_dataframe
                streamlit.rerun()