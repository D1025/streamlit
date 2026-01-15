"""Warstwa logiki aplikacji (obliczenia + operacje na danych).

Plik celowo nie importuje Streamlit, Folium itp., aby logika była
łatwo testowalna i niezależna od UI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd


PointsDF = pd.DataFrame


@dataclass
class CentroidCalculator:
    _pts: List[Tuple[float, float, float, float]] = field(default_factory=list)

    def add(self, x: float, y: float, transport_rate: float, mass: float) -> None:
        self._pts.append((float(x), float(y), float(transport_rate), float(mass)))

    def extend(self, points: Iterable[Sequence[float]]) -> None:
        for p in points:
            self._pts.append((float(p[0]), float(p[1]), float(p[2]), float(p[3])))

    def centroid(self) -> Tuple[float, float]:
        if not self._pts:
            return 0.0, 0.0

        total_weight = 0.0
        weighted_x_sum = 0.0
        weighted_y_sum = 0.0

        for x, y, transport_rate, mass in self._pts:
            w = float(transport_rate) * float(mass)
            total_weight += w
            weighted_x_sum += w * float(x)
            weighted_y_sum += w * float(y)

        if abs(total_weight) < 1e-12:
            n = len(self._pts)
            avg_x = sum(p[0] for p in self._pts) / n
            avg_y = sum(p[1] for p in self._pts) / n
            return float(avg_x), float(avg_y)

        return float(weighted_x_sum / total_weight), float(weighted_y_sum / total_weight)

    def weighted_euclidean_distance_sum(self, centroid_x: float, centroid_y: float) -> float:
        weighted_sum = 0.0
        for x, y, transport_rate, mass in self._pts:
            w = float(transport_rate) * float(mass)
            d = math.sqrt((float(centroid_x) - float(x)) ** 2 + (float(centroid_y) - float(y)) ** 2)
            weighted_sum += w * d
        return float(weighted_sum)


def normalize_column_names(input_dataframe: PointsDF) -> PointsDF:
    df = input_dataframe.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def choose_existing_column(normalized_dataframe: PointsDF, candidate_names: Sequence[str]) -> Optional[str]:
    for name in candidate_names:
        if name in normalized_dataframe.columns:
            return name
    return None


def ensure_points_dataframe(points_dataframe: Optional[PointsDF]) -> PointsDF:
    if points_dataframe is None:
        return pd.DataFrame(columns=["longitude", "latitude", "transport_rate", "mass"])

    df = normalize_column_names(points_dataframe)

    lon_col = choose_existing_column(df, ["longitude", "lon", "x"])
    lat_col = choose_existing_column(df, ["latitude", "lat", "y"])
    tr_col = choose_existing_column(df, ["transport_rate", "transport", "rate", "stawka_transportowa", "stawka", "st"])
    mass_col = choose_existing_column(df, ["mass", "masa", "m"])

    if lon_col is None or lat_col is None:
        if len(df.columns) >= 2:
            inferred_lon = df.columns[0]
            inferred_lat = df.columns[1]
            lon_col = lon_col or inferred_lon
            lat_col = lat_col or inferred_lat

    if tr_col is None:
        df["transport_rate"] = 1.0
        tr_col = "transport_rate"

    if mass_col is None:
        df["mass"] = 1.0
        mass_col = "mass"

    if lon_col is None:
        df["longitude"] = []
        lon_col = "longitude"

    if lat_col is None:
        df["latitude"] = []
        lat_col = "latitude"

    selected = df[[lon_col, lat_col, tr_col, mass_col]].copy()
    selected.columns = ["longitude", "latitude", "transport_rate", "mass"]

    selected["longitude"] = pd.to_numeric(selected["longitude"], errors="coerce")
    selected["latitude"] = pd.to_numeric(selected["latitude"], errors="coerce")
    selected["transport_rate"] = pd.to_numeric(selected["transport_rate"], errors="coerce")
    selected["mass"] = pd.to_numeric(selected["mass"], errors="coerce")

    selected = selected.dropna(subset=["longitude", "latitude"]).reset_index(drop=True)
    selected["transport_rate"] = selected["transport_rate"].fillna(1.0)
    selected["mass"] = selected["mass"].fillna(1.0)

    return selected


def append_point(points_dataframe: Optional[PointsDF], longitude: float, latitude: float, transport_rate: float, mass: float) -> PointsDF:
    df = ensure_points_dataframe(points_dataframe)
    new_row = pd.DataFrame(
        [
            {
                "longitude": float(longitude),
                "latitude": float(latitude),
                "transport_rate": float(transport_rate),
                "mass": float(mass),
            }
        ]
    )
    return pd.concat([df, new_row], ignore_index=True)


def read_points_from_uploaded_file(uploaded_file) -> PointsDF:
    # uploaded_file: Streamlit UploadedFile-like. Trzymamy to poza UI, ale bez
    # zależności na streamlit.
    if uploaded_file is None:
        return pd.DataFrame(columns=["longitude", "latitude", "transport_rate", "mass"])

    uploaded_name = str(getattr(uploaded_file, "name", "")).lower()
    try:
        if uploaded_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_name.endswith(".xlsx") or uploaded_name.endswith(".xls"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
    except Exception:
        return pd.DataFrame(columns=["longitude", "latitude", "transport_rate", "mass"])

    return ensure_points_dataframe(df)


def compute_center_of_gravity(points_dataframe: Optional[PointsDF]) -> Tuple[float, float, float]:
    df = ensure_points_dataframe(points_dataframe)
    calc = CentroidCalculator()
    for _, row in df.iterrows():
        calc.add(row["longitude"], row["latitude"], row["transport_rate"], row["mass"])

    cx, cy = calc.centroid()
    wsum = calc.weighted_euclidean_distance_sum(cx, cy)
    return cx, cy, wsum


def get_map_center(points_dataframe: Optional[PointsDF], centroid_longitude: float, centroid_latitude: float) -> Tuple[float, float]:
    df = ensure_points_dataframe(points_dataframe)
    if len(df) > 0:
        last_lon = float(df.iloc[-1]["longitude"])
        last_lat = float(df.iloc[-1]["latitude"])
        return last_lon, last_lat

    if abs(float(centroid_longitude)) > 1e-9 or abs(float(centroid_latitude)) > 1e-9:
        return float(centroid_longitude), float(centroid_latitude)

    # Warszawa jako domyślna lokalizacja
    return 21.0122, 52.2297


def build_polyline_path(points_dataframe: Optional[PointsDF]) -> List[List[float]]:
    df = ensure_points_dataframe(points_dataframe)
    path: List[List[float]] = []
    for _, row in df.iterrows():
        path.append([float(row["longitude"]), float(row["latitude"])])

    if len(path) >= 3 and path[0] != path[-1]:
        path.append(path[0])

    return path


def compute_point_distances(points_dataframe: Optional[PointsDF], centroid_longitude: float, centroid_latitude: float) -> PointsDF:
    df = ensure_points_dataframe(points_dataframe)
    out = df.copy()

    distances: List[float] = []
    weighted_distances: List[float] = []

    for _, row in out.iterrows():
        d = math.sqrt((float(centroid_longitude) - float(row["longitude"])) ** 2 + (float(centroid_latitude) - float(row["latitude"])) ** 2)
        w = float(row["transport_rate"]) * float(row["mass"])
        distances.append(float(d))
        weighted_distances.append(float(w) * float(d))

    out["euclidean_distance"] = distances
    out["weighted_euclidean_distance"] = weighted_distances
    return out


def extract_marker_positions_from_drawings(all_drawings) -> List[Tuple[float, float]]:
    marker_positions: List[Tuple[float, float]] = []

    if all_drawings is None:
        return marker_positions

    if isinstance(all_drawings, dict):
        drawings = list(all_drawings.values())
    elif isinstance(all_drawings, list):
        drawings = list(all_drawings)
    else:
        return marker_positions

    for feature in drawings:
        if not isinstance(feature, dict):
            continue
        geometry = feature.get("geometry")
        if not isinstance(geometry, dict):
            continue
        if geometry.get("type") != "Point":
            continue
        coords = geometry.get("coordinates")
        if not isinstance(coords, (list, tuple)) or len(coords) < 2:
            continue
        lon, lat = coords[0], coords[1]
        try:
            marker_positions.append((float(lon), float(lat)))
        except Exception:
            continue

    return marker_positions


def synchronize_points_dataframe_with_marker_positions(
    points_dataframe: Optional[PointsDF],
    marker_positions: Iterable[Tuple[float, float]],
    default_transport_rate: float,
    default_mass: float,
) -> PointsDF:
    df = ensure_points_dataframe(points_dataframe)
    marker_positions = list(marker_positions or [])

    if len(marker_positions) == 0:
        return pd.DataFrame(columns=["longitude", "latitude", "transport_rate", "mass"])

    if len(df) == 0:
        rows = [
            {
                "longitude": float(lon),
                "latitude": float(lat),
                "transport_rate": float(default_transport_rate),
                "mass": float(default_mass),
            }
            for lon, lat in marker_positions
        ]
        return ensure_points_dataframe(pd.DataFrame(rows))

    existing_positions = [(float(r["longitude"]), float(r["latitude"])) for _, r in df.iterrows()]
    existing_rows = [
        {
            "longitude": float(r["longitude"]),
            "latitude": float(r["latitude"]),
            "transport_rate": float(r["transport_rate"]),
            "mass": float(r["mass"]),
        }
        for _, r in df.iterrows()
    ]

    all_pairs: List[Tuple[float, int, int]] = []
    for ei, (elon, elat) in enumerate(existing_positions):
        for mi, (mlon, mlat) in enumerate(marker_positions):
            dist = math.sqrt((elon - mlon) ** 2 + (elat - mlat) ** 2)
            all_pairs.append((dist, ei, mi))

    all_pairs.sort(key=lambda item: float(item[0]))

    matched_existing = set()
    matched_markers = set()
    matched_map = {}

    for _, ei, mi in all_pairs:
        if ei in matched_existing or mi in matched_markers:
            continue
        matched_existing.add(ei)
        matched_markers.add(mi)
        matched_map[ei] = mi
        if len(matched_existing) == len(existing_positions) or len(matched_markers) == len(marker_positions):
            break

    updated_rows = []
    for ei in range(len(existing_rows)):
        if ei not in matched_map:
            continue
        mi = matched_map[ei]
        mlon, mlat = marker_positions[mi]
        updated_rows.append(
            {
                "longitude": float(mlon),
                "latitude": float(mlat),
                "transport_rate": float(existing_rows[ei]["transport_rate"]),
                "mass": float(existing_rows[ei]["mass"]),
            }
        )

    for mi in range(len(marker_positions)):
        if mi in matched_markers:
            continue
        mlon, mlat = marker_positions[mi]
        updated_rows.append(
            {
                "longitude": float(mlon),
                "latitude": float(mlat),
                "transport_rate": float(default_transport_rate),
                "mass": float(default_mass),
            }
        )

    return ensure_points_dataframe(pd.DataFrame(updated_rows))


def points_dataframe_signature(points_dataframe: Optional[PointsDF]) -> Tuple[Tuple[float, float, float, float], ...]:
    df = ensure_points_dataframe(points_dataframe)
    items: List[Tuple[float, float, float, float]] = []
    for _, row in df.iterrows():
        items.append(
            (
                round(float(row["longitude"]), 8),
                round(float(row["latitude"]), 8),
                round(float(row["transport_rate"]), 8),
                round(float(row["mass"]), 8),
            )
        )
    return tuple(items)
