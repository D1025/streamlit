from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


PointsDF = pd.DataFrame


@dataclass
class CenterOfGravityDetails:
    centroid_longitude: float
    centroid_latitude: float
    weighted_distance_sum: float
    total_weight: float
    weighted_longitude_sum: float
    weighted_latitude_sum: float
    used_fallback_average: bool
    per_point_breakdown_dataframe: PointsDF


@dataclass
class TopsisDetails:
    ranking_dataframe: PointsDF
    decision_matrix: PointsDF
    normalized_matrix: PointsDF
    weighted_normalized_matrix: PointsDF
    normalized_weights_by_column: Dict[str, float]
    ideal_best_by_column: Dict[str, float]
    ideal_worst_by_column: Dict[str, float]
    distances_to_best: List[float]
    distances_to_worst: List[float]
    topsis_scores: List[float]
    valid_criteria_columns: List[str]


@dataclass
class CentroidCalculator:
    points: List[Tuple[float, float, float, float]] = field(default_factory=list)

    def add(self, longitude: float, latitude: float, transport_rate: float, mass: float) -> None:
        self.points.append((float(longitude), float(latitude), float(transport_rate), float(mass)))

    def extend(self, points: Iterable[Sequence[float]]) -> None:
        for point in points:
            self.points.append((float(point[0]), float(point[1]), float(point[2]), float(point[3])))

    def centroid(self) -> Tuple[float, float, float, float, float, bool]:
        if not self.points:
            return 0.0, 0.0, 0.0, 0.0, 0.0, False

        total_weight = 0.0
        weighted_longitude_sum = 0.0
        weighted_latitude_sum = 0.0

        for longitude, latitude, transport_rate, mass in self.points:
            point_weight = float(transport_rate) * float(mass)
            total_weight += point_weight
            weighted_longitude_sum += point_weight * float(longitude)
            weighted_latitude_sum += point_weight * float(latitude)

        if abs(total_weight) < 1e-12:
            points_count = len(self.points)
            average_longitude = sum(point[0] for point in self.points) / points_count
            average_latitude = sum(point[1] for point in self.points) / points_count
            return float(average_longitude), float(average_latitude), float(total_weight), float(weighted_longitude_sum), float(weighted_latitude_sum), True

        centroid_longitude = weighted_longitude_sum / total_weight
        centroid_latitude = weighted_latitude_sum / total_weight
        return float(centroid_longitude), float(centroid_latitude), float(total_weight), float(weighted_longitude_sum), float(weighted_latitude_sum), False

    def weighted_euclidean_distance_sum(self, centroid_longitude: float, centroid_latitude: float) -> float:
        weighted_distance_sum = 0.0
        for longitude, latitude, transport_rate, mass in self.points:
            point_weight = float(transport_rate) * float(mass)
            euclidean_distance = math.sqrt((float(centroid_longitude) - float(longitude)) ** 2 + (float(centroid_latitude) - float(latitude)) ** 2)
            weighted_distance_sum += point_weight * euclidean_distance
        return float(weighted_distance_sum)


def normalize_column_names(input_dataframe: PointsDF) -> PointsDF:
    normalized_dataframe = input_dataframe.copy()
    normalized_dataframe.columns = [str(column_name).strip().lower() for column_name in normalized_dataframe.columns]
    return normalized_dataframe


def choose_existing_column(normalized_dataframe: PointsDF, candidate_names: Sequence[str]) -> Optional[str]:
    for candidate_name in candidate_names:
        if candidate_name in normalized_dataframe.columns:
            return candidate_name
    return None


def ensure_points_dataframe(points_dataframe: Optional[PointsDF], include_transport_rate_and_mass: bool) -> PointsDF:
    if points_dataframe is None:
        base_dataframe = pd.DataFrame(columns=["longitude", "latitude"])
    else:
        base_dataframe = points_dataframe.copy()

    normalized_dataframe = normalize_column_names(base_dataframe)

    longitude_column = choose_existing_column(normalized_dataframe, ["longitude", "lon", "x"])
    latitude_column = choose_existing_column(normalized_dataframe, ["latitude", "lat", "y"])

    if longitude_column is None or latitude_column is None:
        if len(normalized_dataframe.columns) >= 2:
            inferred_longitude_column = normalized_dataframe.columns[0]
            inferred_latitude_column = normalized_dataframe.columns[1]
            longitude_column = longitude_column or inferred_longitude_column
            latitude_column = latitude_column or inferred_latitude_column

    if longitude_column is None:
        normalized_dataframe["longitude"] = pd.Series(dtype="float64")
        longitude_column = "longitude"

    if latitude_column is None:
        normalized_dataframe["latitude"] = pd.Series(dtype="float64")
        latitude_column = "latitude"

    if longitude_column != "longitude":
        normalized_dataframe["longitude"] = normalized_dataframe[longitude_column]
    if latitude_column != "latitude":
        normalized_dataframe["latitude"] = normalized_dataframe[latitude_column]

    columns_to_drop: List[str] = []
    if longitude_column not in ("longitude", "latitude"):
        columns_to_drop.append(longitude_column)
    if latitude_column not in ("longitude", "latitude"):
        columns_to_drop.append(latitude_column)
    if columns_to_drop:
        normalized_dataframe = normalized_dataframe.drop(columns=columns_to_drop, errors="ignore")

    normalized_dataframe["longitude"] = pd.to_numeric(normalized_dataframe["longitude"], errors="coerce")
    normalized_dataframe["latitude"] = pd.to_numeric(normalized_dataframe["latitude"], errors="coerce")
    normalized_dataframe = normalized_dataframe.dropna(subset=["longitude", "latitude"]).reset_index(drop=True)

    if include_transport_rate_and_mass:
        if "transport_rate" not in normalized_dataframe.columns:
            normalized_dataframe["transport_rate"] = 1.0
        if "mass" not in normalized_dataframe.columns:
            normalized_dataframe["mass"] = 1.0
        normalized_dataframe["transport_rate"] = pd.to_numeric(normalized_dataframe["transport_rate"], errors="coerce").fillna(1.0)
        normalized_dataframe["mass"] = pd.to_numeric(normalized_dataframe["mass"], errors="coerce").fillna(1.0)

    return normalized_dataframe


def append_point(
    points_dataframe: Optional[PointsDF],
    longitude: float,
    latitude: float,
    transport_rate: Optional[float] = None,
    mass: Optional[float] = None,
    additional_columns_values: Optional[Dict[str, float]] = None,
) -> PointsDF:
    existing_points_dataframe = ensure_points_dataframe(points_dataframe, include_transport_rate_and_mass=False)

    new_row_values: Dict[str, float] = {"longitude": float(longitude), "latitude": float(latitude)}

    if transport_rate is not None or "transport_rate" in existing_points_dataframe.columns:
        new_row_values["transport_rate"] = float(transport_rate) if transport_rate is not None else 1.0

    if mass is not None or "mass" in existing_points_dataframe.columns:
        new_row_values["mass"] = float(mass) if mass is not None else 1.0

    normalized_additional_columns_values: Dict[str, float] = {}
    if additional_columns_values:
        normalized_additional_columns_values = {str(key).strip().lower(): float(value) for key, value in additional_columns_values.items()}

    for column_name in existing_points_dataframe.columns:
        if column_name in ("longitude", "latitude"):
            continue
        if column_name in new_row_values:
            continue
        if column_name in normalized_additional_columns_values:
            new_row_values[column_name] = float(normalized_additional_columns_values[column_name])
        else:
            new_row_values[column_name] = 1.0

    for column_name, column_value in normalized_additional_columns_values.items():
        if column_name in ("longitude", "latitude"):
            continue
        if column_name not in new_row_values:
            new_row_values[column_name] = float(column_value)

    new_row_dataframe = pd.DataFrame([new_row_values])
    updated_points_dataframe = pd.concat([existing_points_dataframe, new_row_dataframe], ignore_index=True)
    return updated_points_dataframe


def read_points_from_uploaded_file_with_status(uploaded_file) -> Tuple[PointsDF, str, str]:
    if uploaded_file is None:
        return pd.DataFrame(columns=["longitude", "latitude"]), "Nie wybrano pliku.", "info"

    uploaded_name = str(getattr(uploaded_file, "name", "")).strip()
    uploaded_name_lower = uploaded_name.lower()

    try:
        if uploaded_name_lower.endswith(".csv"):
            uploaded_dataframe = pd.read_csv(uploaded_file)
        elif uploaded_name_lower.endswith(".xlsx") or uploaded_name_lower.endswith(".xls"):
            uploaded_dataframe = pd.read_excel(uploaded_file)
        else:
            uploaded_dataframe = pd.read_csv(uploaded_file)
    except Exception:
        empty_dataframe = pd.DataFrame(columns=["longitude", "latitude"])
        return empty_dataframe, "Nie udało się wczytać pliku. Sprawdź format (CSV lub Excel) i spróbuj ponownie.", "error"

    ensured_points_dataframe = ensure_points_dataframe(uploaded_dataframe, include_transport_rate_and_mass=False)

    if len(ensured_points_dataframe) == 0:
        return ensured_points_dataframe, "Plik wczytano, ale nie znaleziono poprawnych współrzędnych (X i Y).", "warning"

    return ensured_points_dataframe, f"Wczytano punkty z pliku: {uploaded_name}", "success"


def compute_center_of_gravity_details(points_dataframe: Optional[PointsDF]) -> CenterOfGravityDetails:
    ensured_points_dataframe = ensure_points_dataframe(points_dataframe, include_transport_rate_and_mass=True)

    centroid_calculator = CentroidCalculator()
    for _, row in ensured_points_dataframe.iterrows():
        centroid_calculator.add(row["longitude"], row["latitude"], row["transport_rate"], row["mass"])

    centroid_longitude, centroid_latitude, total_weight, weighted_longitude_sum, weighted_latitude_sum, used_fallback_average = centroid_calculator.centroid()
    weighted_distance_sum = centroid_calculator.weighted_euclidean_distance_sum(centroid_longitude, centroid_latitude)

    breakdown_rows: List[Dict[str, float]] = []
    for _, row in ensured_points_dataframe.iterrows():
        transport_rate_value = float(row.get("transport_rate", 1.0))
        mass_value = float(row.get("mass", 1.0))
        point_weight = transport_rate_value * mass_value
        longitude_value = float(row.get("longitude", 0.0))
        latitude_value = float(row.get("latitude", 0.0))
        euclidean_distance = math.sqrt((float(centroid_longitude) - longitude_value) ** 2 + (float(centroid_latitude) - latitude_value) ** 2)
        breakdown_rows.append(
            {
                "longitude": longitude_value,
                "latitude": latitude_value,
                "transport_rate": transport_rate_value,
                "mass": mass_value,
                "point_weight": float(point_weight),
                "weighted_longitude": float(point_weight) * float(longitude_value),
                "weighted_latitude": float(point_weight) * float(latitude_value),
                "euclidean_distance": float(euclidean_distance),
                "weighted_euclidean_distance": float(point_weight) * float(euclidean_distance),
            }
        )

    per_point_breakdown_dataframe = pd.DataFrame(breakdown_rows)

    return CenterOfGravityDetails(
        centroid_longitude=float(centroid_longitude),
        centroid_latitude=float(centroid_latitude),
        weighted_distance_sum=float(weighted_distance_sum),
        total_weight=float(total_weight),
        weighted_longitude_sum=float(weighted_longitude_sum),
        weighted_latitude_sum=float(weighted_latitude_sum),
        used_fallback_average=bool(used_fallback_average),
        per_point_breakdown_dataframe=per_point_breakdown_dataframe,
    )


def get_map_center(points_dataframe: Optional[PointsDF], reference_longitude: float, reference_latitude: float) -> Tuple[float, float]:
    ensured_points_dataframe = ensure_points_dataframe(points_dataframe, include_transport_rate_and_mass=False)

    if len(ensured_points_dataframe) > 0:
        last_point_longitude = float(ensured_points_dataframe.iloc[-1]["longitude"])
        last_point_latitude = float(ensured_points_dataframe.iloc[-1]["latitude"])
        return last_point_longitude, last_point_latitude

    if abs(float(reference_longitude)) > 1e-9 or abs(float(reference_latitude)) > 1e-9:
        return float(reference_longitude), float(reference_latitude)

    return 21.0122, 52.2297


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
        coordinates = geometry.get("coordinates")
        if not isinstance(coordinates, (list, tuple)) or len(coordinates) < 2:
            continue
        longitude_value, latitude_value = coordinates[0], coordinates[1]
        try:
            marker_positions.append((float(longitude_value), float(latitude_value)))
        except Exception:
            continue

    return marker_positions


def synchronize_points_dataframe_with_marker_positions(
    points_dataframe: Optional[PointsDF],
    marker_positions: Iterable[Tuple[float, float]],
    default_values_by_column: Optional[Dict[str, float]] = None,
) -> PointsDF:
    ensured_points_dataframe = ensure_points_dataframe(points_dataframe, include_transport_rate_and_mass=False)
    marker_positions_list = list(marker_positions or [])

    normalized_default_values_by_column: Dict[str, float] = {}
    if default_values_by_column:
        normalized_default_values_by_column = {str(key).strip().lower(): float(value) for key, value in default_values_by_column.items()}

    if len(marker_positions_list) == 0:
        return pd.DataFrame(columns=list(ensured_points_dataframe.columns) if len(ensured_points_dataframe.columns) > 0 else ["longitude", "latitude"])

    non_coordinate_columns = [column_name for column_name in ensured_points_dataframe.columns if column_name not in ("longitude", "latitude")]

    if len(ensured_points_dataframe) == 0:
        created_rows: List[Dict[str, float]] = []
        for marker_longitude, marker_latitude in marker_positions_list:
            created_row: Dict[str, float] = {"longitude": float(marker_longitude), "latitude": float(marker_latitude)}
            for column_name in non_coordinate_columns:
                created_row[column_name] = float(normalized_default_values_by_column.get(column_name, 1.0))
            for column_name, column_value in normalized_default_values_by_column.items():
                if column_name in ("longitude", "latitude"):
                    continue
                if column_name not in created_row:
                    created_row[column_name] = float(column_value)
            created_rows.append(created_row)
        created_dataframe = pd.DataFrame(created_rows)
        return ensure_points_dataframe(created_dataframe, include_transport_rate_and_mass=False)

    existing_positions = [(float(row["longitude"]), float(row["latitude"])) for _, row in ensured_points_dataframe.iterrows()]
    existing_rows: List[Dict[str, float]] = []
    for _, row in ensured_points_dataframe.iterrows():
        existing_row_values: Dict[str, float] = {"longitude": float(row["longitude"]), "latitude": float(row["latitude"])}
        for column_name in non_coordinate_columns:
            existing_row_values[column_name] = row.get(column_name)
        existing_rows.append(existing_row_values)

    all_pairs: List[Tuple[float, int, int]] = []
    for existing_index, (existing_longitude, existing_latitude) in enumerate(existing_positions):
        for marker_index, (marker_longitude, marker_latitude) in enumerate(marker_positions_list):
            distance_value = math.sqrt((existing_longitude - float(marker_longitude)) ** 2 + (existing_latitude - float(marker_latitude)) ** 2)
            all_pairs.append((float(distance_value), int(existing_index), int(marker_index)))

    all_pairs.sort(key=lambda item: float(item[0]))

    matched_existing_indices = set()
    matched_marker_indices = set()
    matched_existing_to_marker: Dict[int, int] = {}

    for _, existing_index, marker_index in all_pairs:
        if existing_index in matched_existing_indices:
            continue
        if marker_index in matched_marker_indices:
            continue
        matched_existing_indices.add(existing_index)
        matched_marker_indices.add(marker_index)
        matched_existing_to_marker[int(existing_index)] = int(marker_index)
        if len(matched_existing_indices) == len(existing_positions) or len(matched_marker_indices) == len(marker_positions_list):
            break

    updated_rows: List[Dict[str, float]] = []
    for existing_index in range(len(existing_rows)):
        if existing_index not in matched_existing_to_marker:
            continue
        marker_index = matched_existing_to_marker[existing_index]
        marker_longitude, marker_latitude = marker_positions_list[marker_index]
        updated_row: Dict[str, float] = {"longitude": float(marker_longitude), "latitude": float(marker_latitude)}
        for column_name in non_coordinate_columns:
            updated_row[column_name] = existing_rows[existing_index].get(column_name)
        updated_rows.append(updated_row)

    for marker_index in range(len(marker_positions_list)):
        if marker_index in matched_marker_indices:
            continue
        marker_longitude, marker_latitude = marker_positions_list[marker_index]
        new_row: Dict[str, float] = {"longitude": float(marker_longitude), "latitude": float(marker_latitude)}
        for column_name in non_coordinate_columns:
            new_row[column_name] = float(normalized_default_values_by_column.get(column_name, 1.0))
        for column_name, column_value in normalized_default_values_by_column.items():
            if column_name in ("longitude", "latitude"):
                continue
            if column_name not in new_row:
                new_row[column_name] = float(column_value)
        updated_rows.append(new_row)

    updated_dataframe = pd.DataFrame(updated_rows)

    ordered_columns = ["longitude", "latitude"]
    ordered_columns.extend([column_name for column_name in updated_dataframe.columns if column_name not in ("longitude", "latitude")])

    updated_dataframe = updated_dataframe[ordered_columns]
    return ensure_points_dataframe(updated_dataframe, include_transport_rate_and_mass=False)


def points_dataframe_signature(points_dataframe: Optional[PointsDF]) -> Tuple[Tuple[str, ...], Tuple[Tuple[str, ...], ...]]:
    ensured_points_dataframe = ensure_points_dataframe(points_dataframe, include_transport_rate_and_mass=False)

    sorted_column_names = sorted([str(column_name) for column_name in ensured_points_dataframe.columns])
    signature_rows: List[Tuple[str, ...]] = []

    for _, row in ensured_points_dataframe.iterrows():
        row_signature_values: List[str] = []
        for column_name in sorted_column_names:
            cell_value = row.get(column_name)
            if cell_value is None or (isinstance(cell_value, float) and math.isnan(cell_value)):
                row_signature_values.append("")
            elif isinstance(cell_value, (int, float)):
                row_signature_values.append(str(round(float(cell_value), 8)))
            else:
                row_signature_values.append(str(cell_value))
        signature_rows.append(tuple(row_signature_values))

    return tuple(sorted_column_names), tuple(signature_rows)


def get_topsis_candidate_criteria_columns(points_dataframe: Optional[PointsDF]) -> List[str]:
    ensured_points_dataframe = ensure_points_dataframe(points_dataframe, include_transport_rate_and_mass=False)

    excluded_columns = {
        "longitude",
        "latitude",
        "transport_rate",
        "mass",
        "euclidean_distance",
        "weighted_euclidean_distance",
        "topsis_score",
        "topsis_rank",
    }

    candidate_columns: List[str] = []
    for column_name in ensured_points_dataframe.columns:
        normalized_column_name = str(column_name).strip().lower()
        if normalized_column_name in excluded_columns:
            continue
        candidate_columns.append(str(column_name))

    numeric_candidate_columns: List[str] = []
    for column_name in candidate_columns:
        column_values = pd.to_numeric(ensured_points_dataframe[column_name], errors="coerce")
        non_missing_count = int(column_values.notna().sum())
        if non_missing_count > 0:
            numeric_candidate_columns.append(str(column_name))

    return sorted(numeric_candidate_columns)


def compute_topsis_details(
    points_dataframe: Optional[PointsDF],
    criteria_columns: Sequence[str],
    criteria_weights_by_name: Optional[Dict[str, float]] = None,
    criteria_impacts_by_name: Optional[Dict[str, str]] = None,
) -> TopsisDetails:
    ensured_points_dataframe = ensure_points_dataframe(points_dataframe, include_transport_rate_and_mass=False).copy()
    normalized_criteria_columns = [str(column_name).strip().lower() for column_name in criteria_columns if str(column_name).strip()]

    normalized_dataframe = normalize_column_names(ensured_points_dataframe)
    ensured_points_dataframe = normalized_dataframe.reset_index(drop=True)

    if len(ensured_points_dataframe) == 0:
        empty_dataframe = ensured_points_dataframe.copy()
        empty_dataframe["topsis_score"] = pd.Series(dtype="float64")
        empty_dataframe["topsis_rank"] = pd.Series(dtype="int64")
        return TopsisDetails(
            ranking_dataframe=empty_dataframe,
            decision_matrix=pd.DataFrame(),
            normalized_matrix=pd.DataFrame(),
            weighted_normalized_matrix=pd.DataFrame(),
            normalized_weights_by_column={},
            ideal_best_by_column={},
            ideal_worst_by_column={},
            distances_to_best=[],
            distances_to_worst=[],
            topsis_scores=[],
            valid_criteria_columns=[],
        )

    valid_criteria_columns = [column_name for column_name in normalized_criteria_columns if column_name in ensured_points_dataframe.columns]

    if len(valid_criteria_columns) == 0:
        fallback_ranking_dataframe = ensured_points_dataframe.copy()
        fallback_ranking_dataframe["topsis_score"] = 0.0
        fallback_ranking_dataframe["topsis_rank"] = list(range(1, len(fallback_ranking_dataframe) + 1))
        return TopsisDetails(
            ranking_dataframe=fallback_ranking_dataframe,
            decision_matrix=pd.DataFrame(index=fallback_ranking_dataframe.index),
            normalized_matrix=pd.DataFrame(index=fallback_ranking_dataframe.index),
            weighted_normalized_matrix=pd.DataFrame(index=fallback_ranking_dataframe.index),
            normalized_weights_by_column={},
            ideal_best_by_column={},
            ideal_worst_by_column={},
            distances_to_best=[0.0 for _ in range(len(fallback_ranking_dataframe))],
            distances_to_worst=[0.0 for _ in range(len(fallback_ranking_dataframe))],
            topsis_scores=[0.0 for _ in range(len(fallback_ranking_dataframe))],
            valid_criteria_columns=[],
        )

    weights_by_name: Dict[str, float] = {}
    if criteria_weights_by_name:
        weights_by_name = {str(key).strip().lower(): float(value) for key, value in criteria_weights_by_name.items()}

    impacts_by_name: Dict[str, str] = {}
    if criteria_impacts_by_name:
        impacts_by_name = {str(key).strip().lower(): str(value).strip().lower() for key, value in criteria_impacts_by_name.items()}

    decision_matrix = ensured_points_dataframe[valid_criteria_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    raw_weights: List[float] = []
    for column_name in valid_criteria_columns:
        raw_weights.append(float(weights_by_name.get(column_name, 1.0)))

    total_weight = float(sum(raw_weights))
    if abs(total_weight) < 1e-12:
        normalized_weights = [1.0 / float(len(valid_criteria_columns)) for _ in valid_criteria_columns]
    else:
        normalized_weights = [float(weight_value) / total_weight for weight_value in raw_weights]

    normalized_weights_by_column: Dict[str, float] = {}
    for column_index, column_name in enumerate(valid_criteria_columns):
        normalized_weights_by_column[str(column_name)] = float(normalized_weights[column_index])

    normalized_matrix = decision_matrix.copy()
    for column_name in valid_criteria_columns:
        column_values = decision_matrix[column_name].astype(float)
        denominator = float(math.sqrt(float((column_values ** 2).sum())))
        if abs(denominator) < 1e-12:
            normalized_matrix[column_name] = 0.0
        else:
            normalized_matrix[column_name] = column_values / denominator

    weighted_normalized_matrix = normalized_matrix.copy()
    for column_index, column_name in enumerate(valid_criteria_columns):
        weighted_normalized_matrix[column_name] = weighted_normalized_matrix[column_name].astype(float) * float(normalized_weights[column_index])

    ideal_best_by_column: Dict[str, float] = {}
    ideal_worst_by_column: Dict[str, float] = {}

    for column_name in valid_criteria_columns:
        impact_value = impacts_by_name.get(column_name, "benefit")
        column_values = weighted_normalized_matrix[column_name].astype(float)
        if impact_value == "cost":
            ideal_best_by_column[column_name] = float(column_values.min())
            ideal_worst_by_column[column_name] = float(column_values.max())
        else:
            ideal_best_by_column[column_name] = float(column_values.max())
            ideal_worst_by_column[column_name] = float(column_values.min())

    distances_to_best: List[float] = []
    distances_to_worst: List[float] = []

    for _, row in weighted_normalized_matrix.iterrows():
        best_distance_sum = 0.0
        worst_distance_sum = 0.0
        for column_name in valid_criteria_columns:
            value = float(row[column_name])
            best_distance_sum += (value - float(ideal_best_by_column[column_name])) ** 2
            worst_distance_sum += (value - float(ideal_worst_by_column[column_name])) ** 2
        distances_to_best.append(float(math.sqrt(best_distance_sum)))
        distances_to_worst.append(float(math.sqrt(worst_distance_sum)))

    topsis_scores: List[float] = []
    for best_distance, worst_distance in zip(distances_to_best, distances_to_worst):
        denominator = float(best_distance) + float(worst_distance)
        if abs(denominator) < 1e-12:
            topsis_scores.append(0.0)
        else:
            topsis_scores.append(float(worst_distance) / denominator)

    sorting_helper_dataframe = pd.DataFrame(
        {
            "row_index": list(range(len(ensured_points_dataframe))),
            "topsis_score": [float(score_value) for score_value in topsis_scores],
        }
    )
    sorted_row_indices = list(sorting_helper_dataframe.sort_values(by=["topsis_score"], ascending=False)["row_index"].astype(int).tolist())

    ranking_dataframe = ensured_points_dataframe.iloc[sorted_row_indices].reset_index(drop=True)
    ranking_dataframe["topsis_score"] = [float(topsis_scores[row_index]) for row_index in sorted_row_indices]
    ranking_dataframe["topsis_rank"] = list(range(1, len(ranking_dataframe) + 1))

    decision_matrix = decision_matrix.iloc[sorted_row_indices].reset_index(drop=True)
    normalized_matrix = normalized_matrix.iloc[sorted_row_indices].reset_index(drop=True)
    weighted_normalized_matrix = weighted_normalized_matrix.iloc[sorted_row_indices].reset_index(drop=True)

    distances_to_best_sorted = [float(distances_to_best[row_index]) for row_index in sorted_row_indices]
    distances_to_worst_sorted = [float(distances_to_worst[row_index]) for row_index in sorted_row_indices]
    topsis_scores_sorted = [float(topsis_scores[row_index]) for row_index in sorted_row_indices]

    return TopsisDetails(
        ranking_dataframe=ranking_dataframe,
        decision_matrix=decision_matrix,
        normalized_matrix=normalized_matrix,
        weighted_normalized_matrix=weighted_normalized_matrix,
        normalized_weights_by_column=normalized_weights_by_column,
        ideal_best_by_column={str(key): float(value) for key, value in ideal_best_by_column.items()},
        ideal_worst_by_column={str(key): float(value) for key, value in ideal_worst_by_column.items()},
        distances_to_best=distances_to_best_sorted,
        distances_to_worst=distances_to_worst_sorted,
        topsis_scores=topsis_scores_sorted,
        valid_criteria_columns=[str(column_name) for column_name in valid_criteria_columns],
    )
