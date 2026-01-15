from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium

from logic import (
    append_point,
    compute_center_of_gravity,
    compute_point_distances,
    compute_topsis_ranking,
    ensure_points_dataframe,
    extract_marker_positions_from_drawings,
    get_map_center,
    get_topsis_candidate_criteria_columns,
    points_dataframe_signature,
    read_points_from_uploaded_file,
    synchronize_points_dataframe_with_marker_positions,
)


def init_session_state() -> None:
    if "points_dataframe" not in st.session_state:
        st.session_state["points_dataframe"] = pd.DataFrame(columns=["longitude", "latitude"])

    if "calculation_method" not in st.session_state:
        st.session_state["calculation_method"] = "Środek ciężkości"

    if "map_default_transport_rate" not in st.session_state:
        st.session_state["map_default_transport_rate"] = 1.0

    if "map_default_mass" not in st.session_state:
        st.session_state["map_default_mass"] = 1.0

    if "topsis_selected_criteria_columns" not in st.session_state:
        st.session_state["topsis_selected_criteria_columns"] = []

    if "topsis_criteria_weights" not in st.session_state:
        st.session_state["topsis_criteria_weights"] = {}

    if "topsis_criteria_impacts" not in st.session_state:
        st.session_state["topsis_criteria_impacts"] = {}

    if "topsis_default_values_by_criteria" not in st.session_state:
        st.session_state["topsis_default_values_by_criteria"] = {}

    if "map_center_latitude" not in st.session_state:
        st.session_state["map_center_latitude"] = 52.2297

    if "map_center_longitude" not in st.session_state:
        st.session_state["map_center_longitude"] = 21.0122

    if "map_zoom_level" not in st.session_state:
        st.session_state["map_zoom_level"] = 11

    if "map_marker_positions_snapshot" not in st.session_state:
        st.session_state["map_marker_positions_snapshot"] = ()

    if "interactive_folium_map_key_version" not in st.session_state:
        st.session_state["interactive_folium_map_key_version"] = 0


def get_interactive_folium_map_key() -> str:
    return f"interactive_folium_map_{int(st.session_state['interactive_folium_map_key_version'])}"


def bump_interactive_folium_map_key() -> None:
    st.session_state["interactive_folium_map_key_version"] = int(st.session_state["interactive_folium_map_key_version"]) + 1


def update_topsis_state_for_available_criteria(available_criteria_columns: List[str]) -> None:
    selected_criteria_columns = list(st.session_state.get("topsis_selected_criteria_columns", []))
    selected_criteria_columns = [str(column_name).strip().lower() for column_name in selected_criteria_columns if str(column_name).strip()]
    available_criteria_columns = [str(column_name).strip().lower() for column_name in available_criteria_columns if str(column_name).strip()]

    selected_criteria_columns = [column_name for column_name in selected_criteria_columns if column_name in available_criteria_columns]

    if len(selected_criteria_columns) == 0 and len(available_criteria_columns) > 0:
        selected_criteria_columns = list(available_criteria_columns)

    st.session_state["topsis_selected_criteria_columns"] = selected_criteria_columns

    criteria_weights: Dict[str, float] = dict(st.session_state.get("topsis_criteria_weights", {}))
    criteria_impacts: Dict[str, str] = dict(st.session_state.get("topsis_criteria_impacts", {}))
    default_values_by_criteria: Dict[str, float] = dict(st.session_state.get("topsis_default_values_by_criteria", {}))

    updated_criteria_weights: Dict[str, float] = {}
    updated_criteria_impacts: Dict[str, str] = {}
    updated_default_values_by_criteria: Dict[str, float] = {}

    for column_name in selected_criteria_columns:
        updated_criteria_weights[column_name] = float(criteria_weights.get(column_name, 1.0))
        updated_criteria_impacts[column_name] = str(criteria_impacts.get(column_name, "benefit")).strip().lower()
        updated_default_values_by_criteria[column_name] = float(default_values_by_criteria.get(column_name, 1.0))

    st.session_state["topsis_criteria_weights"] = updated_criteria_weights
    st.session_state["topsis_criteria_impacts"] = updated_criteria_impacts
    st.session_state["topsis_default_values_by_criteria"] = updated_default_values_by_criteria


def render_controls(calculation_method: str) -> None:
    st.subheader("Dane wejściowe")

    uploaded_file = st.file_uploader(
        "Wczytaj punkty z pliku (CSV/XLSX).",
        type=["csv", "xlsx", "xls"],
    )
    if uploaded_file is not None:
        uploaded_points_dataframe = read_points_from_uploaded_file(uploaded_file)
        if len(uploaded_points_dataframe) > 0:
            st.session_state["points_dataframe"] = uploaded_points_dataframe
            st.session_state["map_marker_positions_snapshot"] = ()
            bump_interactive_folium_map_key()

    if calculation_method == "TOPSIS":
        points_dataframe = ensure_points_dataframe(st.session_state["points_dataframe"], include_transport_rate_and_mass=False)

        st.divider()
        st.subheader("Kryteria TOPSIS")

        criterion_column_name = st.text_input("Nazwa kryterium (kolumny)", value="")
        criterion_default_value = st.number_input("Domyślna wartość kryterium", value=1.0, format="%.6f")
        if st.button("Dodaj kryterium", width="stretch"):
            normalized_criterion_column_name = str(criterion_column_name).strip().lower()
            if normalized_criterion_column_name:
                updated_points_dataframe = points_dataframe.copy()
                if normalized_criterion_column_name not in updated_points_dataframe.columns:
                    updated_points_dataframe[normalized_criterion_column_name] = float(criterion_default_value)
                    st.session_state["points_dataframe"] = updated_points_dataframe
                    st.session_state["map_marker_positions_snapshot"] = ()
                    bump_interactive_folium_map_key()

        points_dataframe = ensure_points_dataframe(st.session_state["points_dataframe"], include_transport_rate_and_mass=False)

        available_criteria_columns = get_topsis_candidate_criteria_columns(points_dataframe)
        update_topsis_state_for_available_criteria(available_criteria_columns)

        selected_criteria_columns = st.multiselect(
            "Wybierz kryteria do obliczeń",
            options=available_criteria_columns,
            default=list(st.session_state.get("topsis_selected_criteria_columns", [])),
            key="topsis_selected_criteria_columns_widget",
        )
        st.session_state["topsis_selected_criteria_columns"] = [str(column_name).strip().lower() for column_name in selected_criteria_columns if str(column_name).strip()]
        update_topsis_state_for_available_criteria(available_criteria_columns)

        st.divider()
        st.subheader("Tabela punktów (alternatywy)")

        data_editor_function = getattr(st, "data_editor", None)
        if data_editor_function is not None:
            previous_signature = points_dataframe_signature(points_dataframe)
            edited_points_dataframe = data_editor_function(points_dataframe, num_rows="dynamic", width="stretch")
            edited_points_dataframe = ensure_points_dataframe(edited_points_dataframe, include_transport_rate_and_mass=False)
            edited_signature = points_dataframe_signature(edited_points_dataframe)
            if edited_signature != previous_signature:
                st.session_state["points_dataframe"] = edited_points_dataframe
                st.session_state["map_marker_positions_snapshot"] = ()
                bump_interactive_folium_map_key()
        else:
            st.dataframe(points_dataframe, width="stretch")

        if st.button("Wyczyść wszystkie punkty", width="stretch"):
            st.session_state["points_dataframe"] = pd.DataFrame(columns=["longitude", "latitude"])
            st.session_state["map_marker_positions_snapshot"] = ()
            bump_interactive_folium_map_key()

        points_dataframe = ensure_points_dataframe(st.session_state["points_dataframe"], include_transport_rate_and_mass=False)
        available_criteria_columns = get_topsis_candidate_criteria_columns(points_dataframe)
        update_topsis_state_for_available_criteria(available_criteria_columns)

        selected_criteria_columns = list(st.session_state.get("topsis_selected_criteria_columns", []))
        criteria_weights: Dict[str, float] = dict(st.session_state.get("topsis_criteria_weights", {}))
        criteria_impacts: Dict[str, str] = dict(st.session_state.get("topsis_criteria_impacts", {}))
        default_values_by_criteria: Dict[str, float] = dict(st.session_state.get("topsis_default_values_by_criteria", {}))

        st.divider()
        st.subheader("Konfiguracja TOPSIS")

        for criterion_name in selected_criteria_columns:
            current_weight_value = float(criteria_weights.get(criterion_name, 1.0))
            current_impact_value = str(criteria_impacts.get(criterion_name, "benefit")).strip().lower()
            current_default_value = float(default_values_by_criteria.get(criterion_name, 1.0))

            criteria_weights[criterion_name] = st.number_input(
                f"Waga: {criterion_name}",
                value=float(current_weight_value),
                min_value=0.0,
                format="%.6f",
                key=f"topsis_weight_{criterion_name}",
            )

            impact_index = 0 if current_impact_value != "cost" else 1
            selected_impact = st.selectbox(
                f"Typ: {criterion_name}",
                options=["benefit", "cost"],
                index=int(impact_index),
                key=f"topsis_impact_{criterion_name}",
            )
            criteria_impacts[criterion_name] = str(selected_impact).strip().lower()

            default_values_by_criteria[criterion_name] = st.number_input(
                f"Domyślna wartość (mapa): {criterion_name}",
                value=float(current_default_value),
                format="%.6f",
                key=f"topsis_default_{criterion_name}",
            )

        st.session_state["topsis_criteria_weights"] = {str(key).strip().lower(): float(value) for key, value in criteria_weights.items()}
        st.session_state["topsis_criteria_impacts"] = {str(key).strip().lower(): str(value).strip().lower() for key, value in criteria_impacts.items()}
        st.session_state["topsis_default_values_by_criteria"] = {str(key).strip().lower(): float(value) for key, value in default_values_by_criteria.items()}

        st.divider()
        st.subheader("Dodaj punkt ręcznie")

        manual_longitude = st.number_input("X (longitude)", value=21.012200, format="%.6f", key="topsis_manual_longitude")
        manual_latitude = st.number_input("Y (latitude)", value=52.229700, format="%.6f", key="topsis_manual_latitude")

        manual_criteria_values: Dict[str, float] = {}
        for criterion_name in selected_criteria_columns:
            manual_criteria_values[criterion_name] = st.number_input(
                f"Wartość: {criterion_name}",
                value=float(st.session_state["topsis_default_values_by_criteria"].get(criterion_name, 1.0)),
                format="%.6f",
                key=f"topsis_manual_{criterion_name}",
            )

        if st.button("Dodaj punkt", width="stretch", key="topsis_add_point_button"):
            st.session_state["points_dataframe"] = append_point(
                st.session_state["points_dataframe"],
                manual_longitude,
                manual_latitude,
                transport_rate=None,
                mass=None,
                additional_columns_values=manual_criteria_values,
            )
            st.session_state["map_marker_positions_snapshot"] = ()
            bump_interactive_folium_map_key()

        st.divider()
        st.subheader("Wynik")

        if len(points_dataframe) == 0 or len(selected_criteria_columns) == 0:
            st.info("Dodaj co najmniej jeden punkt oraz wybierz co najmniej jedno kryterium.")
        else:
            ranking_dataframe = compute_topsis_ranking(
                points_dataframe,
                criteria_columns=selected_criteria_columns,
                criteria_weights_by_name=st.session_state["topsis_criteria_weights"],
                criteria_impacts_by_name=st.session_state["topsis_criteria_impacts"],
            )

            if len(ranking_dataframe) == 0:
                st.info("Brak danych do obliczeń.")
            else:
                best_row = ranking_dataframe.iloc[0]
                st.metric("TOPSIS score (najlepszy)", f"{float(best_row.get('topsis_score', 0.0)):.6f}")
                st.metric("X (longitude)", f"{float(best_row.get('longitude', 0.0)):.6f}")
                st.metric("Y (latitude)", f"{float(best_row.get('latitude', 0.0)):.6f}")
                st.subheader("Ranking")
                st.dataframe(ranking_dataframe, width="stretch")

        return

    points_dataframe = ensure_points_dataframe(st.session_state["points_dataframe"], include_transport_rate_and_mass=True)

    display_dataframe = points_dataframe.copy()
    if "transport_rate" not in display_dataframe.columns:
        display_dataframe["transport_rate"] = 1.0
    if "mass" not in display_dataframe.columns:
        display_dataframe["mass"] = 1.0
    display_dataframe = display_dataframe[["longitude", "latitude", "transport_rate", "mass"]]

    data_editor_function = getattr(st, "data_editor", None)
    if data_editor_function is not None:
        previous_signature = points_dataframe_signature(display_dataframe)
        edited_display_dataframe = data_editor_function(display_dataframe, num_rows="dynamic", width="stretch")
        edited_display_dataframe = ensure_points_dataframe(edited_display_dataframe, include_transport_rate_and_mass=True)
        edited_signature = points_dataframe_signature(edited_display_dataframe)
        if edited_signature != previous_signature:
            full_points_dataframe = ensure_points_dataframe(st.session_state["points_dataframe"], include_transport_rate_and_mass=True)
            for column_name in ["longitude", "latitude", "transport_rate", "mass"]:
                full_points_dataframe[column_name] = edited_display_dataframe[column_name]
            st.session_state["points_dataframe"] = full_points_dataframe
            st.session_state["map_marker_positions_snapshot"] = ()
            bump_interactive_folium_map_key()
    else:
        st.dataframe(display_dataframe, width="stretch")

    st.divider()
    st.subheader("Dodaj punkt ręcznie")

    manual_longitude = st.number_input("X (longitude)", value=21.012200, format="%.6f", key="centroid_manual_longitude")
    manual_latitude = st.number_input("Y (latitude)", value=52.229700, format="%.6f", key="centroid_manual_latitude")
    manual_transport_rate = st.number_input("ST (stawka transportowa)", value=1.0, min_value=0.0, format="%.6f", key="centroid_manual_transport_rate")
    manual_mass = st.number_input("M (masa)", value=1.0, min_value=0.0, format="%.6f", key="centroid_manual_mass")

    if st.button("Dodaj punkt", width="stretch", key="centroid_add_point_button"):
        st.session_state["points_dataframe"] = append_point(
            st.session_state["points_dataframe"],
            manual_longitude,
            manual_latitude,
            transport_rate=manual_transport_rate,
            mass=manual_mass,
            additional_columns_values=None,
        )
        st.session_state["map_marker_positions_snapshot"] = ()
        bump_interactive_folium_map_key()

    if st.button("Wyczyść wszystkie punkty", width="stretch", key="centroid_clear_points_button"):
        st.session_state["points_dataframe"] = pd.DataFrame(columns=["longitude", "latitude", "transport_rate", "mass"])
        st.session_state["map_marker_positions_snapshot"] = ()
        bump_interactive_folium_map_key()

    points_dataframe = ensure_points_dataframe(st.session_state["points_dataframe"], include_transport_rate_and_mass=True)
    centroid_longitude, centroid_latitude, weighted_distance_sum = compute_center_of_gravity(points_dataframe)

    st.divider()
    st.subheader("Wynik")

    if len(points_dataframe) == 0:
        st.info("Dodaj co najmniej jeden punkt, aby wyliczyć X i Y.")
    else:
        st.metric("X (longitude)", f"{centroid_longitude:.6f}")
        st.metric("Y (latitude)", f"{centroid_latitude:.6f}")
        st.metric("Suma ważona odległości euklidesowych", f"{weighted_distance_sum:.6f}")

        distances_dataframe = compute_point_distances(points_dataframe, centroid_longitude, centroid_latitude)
        st.subheader("Odległości (metryka euklidesowa)")
        st.dataframe(distances_dataframe, width="stretch")

    st.divider()
    st.subheader("Domyślne wartości dla punktów dodanych z mapy")

    st.session_state["map_default_transport_rate"] = st.number_input(
        "Domyślne ST",
        value=float(st.session_state["map_default_transport_rate"]),
        min_value=0.0,
        format="%.6f",
        key="centroid_map_default_transport_rate",
    )
    st.session_state["map_default_mass"] = st.number_input(
        "Domyślne M",
        value=float(st.session_state["map_default_mass"]),
        min_value=0.0,
        format="%.6f",
        key="centroid_map_default_mass",
    )


def build_default_values_by_column_for_map(calculation_method: str, points_dataframe: pd.DataFrame) -> Dict[str, float]:
    ensured_points_dataframe = ensure_points_dataframe(points_dataframe, include_transport_rate_and_mass=False)
    default_values_by_column: Dict[str, float] = {}

    non_coordinate_columns = [column_name for column_name in ensured_points_dataframe.columns if column_name not in ("longitude", "latitude")]

    if calculation_method == "TOPSIS":
        selected_criteria_columns = list(st.session_state.get("topsis_selected_criteria_columns", []))
        default_values_by_criteria = dict(st.session_state.get("topsis_default_values_by_criteria", {}))
        for column_name in non_coordinate_columns:
            normalized_column_name = str(column_name).strip().lower()
            if normalized_column_name in selected_criteria_columns:
                default_values_by_column[normalized_column_name] = float(default_values_by_criteria.get(normalized_column_name, 1.0))
            else:
                default_values_by_column[normalized_column_name] = 1.0
        return default_values_by_column

    default_values_by_column["transport_rate"] = float(st.session_state.get("map_default_transport_rate", 1.0))
    default_values_by_column["mass"] = float(st.session_state.get("map_default_mass", 1.0))
    return default_values_by_column


def compute_result_marker_for_map(calculation_method: str, points_dataframe: pd.DataFrame) -> Tuple[float, float, str]:
    ensured_points_dataframe = ensure_points_dataframe(points_dataframe, include_transport_rate_and_mass=False)

    if calculation_method == "TOPSIS":
        available_criteria_columns = get_topsis_candidate_criteria_columns(ensured_points_dataframe)
        update_topsis_state_for_available_criteria(available_criteria_columns)
        selected_criteria_columns = list(st.session_state.get("topsis_selected_criteria_columns", []))
        if len(ensured_points_dataframe) == 0 or len(selected_criteria_columns) == 0:
            return 0.0, 0.0, ""
        ranking_dataframe = compute_topsis_ranking(
            ensured_points_dataframe,
            criteria_columns=selected_criteria_columns,
            criteria_weights_by_name=st.session_state.get("topsis_criteria_weights", {}),
            criteria_impacts_by_name=st.session_state.get("topsis_criteria_impacts", {}),
        )
        if len(ranking_dataframe) == 0:
            return 0.0, 0.0, ""
        best_row = ranking_dataframe.iloc[0]
        best_longitude = float(best_row.get("longitude", 0.0))
        best_latitude = float(best_row.get("latitude", 0.0))
        best_score = float(best_row.get("topsis_score", 0.0))
        tooltip_text = f"Wynik TOPSIS: score={best_score:.6f}, Y={best_latitude:.6f}, X={best_longitude:.6f}"
        return best_longitude, best_latitude, tooltip_text

    centroid_longitude, centroid_latitude, _ = compute_center_of_gravity(ensure_points_dataframe(ensured_points_dataframe, include_transport_rate_and_mass=True))
    tooltip_text = f"Wynik: Y={centroid_latitude:.6f}, X={centroid_longitude:.6f}"
    return float(centroid_longitude), float(centroid_latitude), tooltip_text


def render_map(calculation_method: str) -> None:
    current_map_key = get_interactive_folium_map_key()
    previous_map_interaction = st.session_state.get(current_map_key)

    if isinstance(previous_map_interaction, dict):
        returned_center = previous_map_interaction.get("center")
        returned_zoom = previous_map_interaction.get("zoom")

        if isinstance(returned_center, dict) and "lat" in returned_center and "lng" in returned_center:
            try:
                st.session_state["map_center_latitude"] = float(returned_center["lat"])
                st.session_state["map_center_longitude"] = float(returned_center["lng"])
            except Exception:
                pass

        if returned_zoom is not None:
            try:
                st.session_state["map_zoom_level"] = int(returned_zoom)
            except Exception:
                pass

        last_active_drawing = previous_map_interaction.get("last_active_drawing")
        if isinstance(last_active_drawing, dict):
            marker_positions = extract_marker_positions_from_drawings(previous_map_interaction.get("all_drawings"))
            marker_positions_signature = tuple((round(float(lon), 8), round(float(lat), 8)) for lon, lat in marker_positions)

            previous_snapshot = tuple(st.session_state.get("map_marker_positions_snapshot", ()))
            if marker_positions_signature != previous_snapshot:
                current_points_dataframe = ensure_points_dataframe(st.session_state["points_dataframe"], include_transport_rate_and_mass=False)
                default_values_by_column = build_default_values_by_column_for_map(calculation_method, current_points_dataframe)

                synchronized_points_dataframe = synchronize_points_dataframe_with_marker_positions(
                    current_points_dataframe,
                    marker_positions,
                    default_values_by_column=default_values_by_column,
                )

                st.session_state["points_dataframe"] = synchronized_points_dataframe
                st.session_state["map_marker_positions_snapshot"] = marker_positions_signature

    points_dataframe = ensure_points_dataframe(st.session_state["points_dataframe"], include_transport_rate_and_mass=False)

    result_longitude, result_latitude, result_tooltip_text = compute_result_marker_for_map(calculation_method, points_dataframe)

    map_center_longitude, map_center_latitude = get_map_center(points_dataframe, result_longitude, result_latitude)

    if "map_center_latitude" not in st.session_state or "map_center_longitude" not in st.session_state:
        st.session_state["map_center_latitude"] = float(map_center_latitude)
        st.session_state["map_center_longitude"] = float(map_center_longitude)

    if "map_zoom_level" not in st.session_state:
        st.session_state["map_zoom_level"] = 11

    st.subheader("Mapa")
    st.caption(
        "Dodawanie: wybierz narzędzie markera. Edycja: użyj narzędzia Edit i przeciągnij marker. "
        "Usuwanie: użyj narzędzia Delete i kliknij marker."
    )

    folium_map = folium.Map(
        location=[float(st.session_state["map_center_latitude"]), float(st.session_state["map_center_longitude"])],
        zoom_start=int(st.session_state["map_zoom_level"]),
        control_scale=True,
    )

    editable_feature_group = folium.FeatureGroup(name="Punkty")
    editable_feature_group.add_to(folium_map)

    for _, row in points_dataframe.iterrows():
        folium.Marker(
            location=[float(row["latitude"]), float(row["longitude"])],
            tooltip=f"Y={float(row['latitude']):.6f}, X={float(row['longitude']):.6f}",
        ).add_to(editable_feature_group)

    Draw(
        export=False,
        feature_group=editable_feature_group,
        draw_options={
            "polyline": False,
            "polygon": False,
            "rectangle": False,
            "circle": False,
            "circlemarker": False,
            "marker": True,
        },
        edit_options={
            "edit": True,
            "remove": True,
        },
    ).add_to(folium_map)

    if len(points_dataframe) > 0 and (abs(float(result_longitude)) > 1e-12 or abs(float(result_latitude)) > 1e-12):
        folium.Marker(
            location=[float(result_latitude), float(result_longitude)],
            tooltip=result_tooltip_text,
            icon=folium.Icon(color="red"),
        ).add_to(folium_map)

    st_folium(
        folium_map,
        height=600,
        use_container_width=True,
        key=current_map_key,
        returned_objects=["all_drawings", "last_active_drawing", "center", "zoom"],
        center=[float(st.session_state["map_center_latitude"]), float(st.session_state["map_center_longitude"])],
        zoom=int(st.session_state["map_zoom_level"]),
    )


def run_app() -> None:
    st.set_page_config(page_title="Środek ciężkości logistyczny", layout="wide")

    init_session_state()

    st.title("Lokalizacja punktu logistycznego metodą środka ciężkości / TOPSIS")
    st.write(
        "Środek ciężkości: X = Σ(ST·M·X) / Σ(ST·M) oraz Y = Σ(ST·M·Y) / Σ(ST·M). "
        "TOPSIS: ranking alternatyw na podstawie wybranych kryteriów (benefit/cost) i wag."
    )

    calculation_method = st.radio(
        "Metoda",
        options=["Środek ciężkości", "TOPSIS"],
        horizontal=True,
        key="calculation_method_radio",
        index=0 if str(st.session_state.get("calculation_method", "Środek ciężkości")) != "TOPSIS" else 1,
    )
    if calculation_method != st.session_state.get("calculation_method"):
        st.session_state["calculation_method"] = str(calculation_method)
        st.session_state["map_marker_positions_snapshot"] = ()
        bump_interactive_folium_map_key()

    control_column, map_column = st.columns([1, 2], gap="large")

    with map_column:
        render_map(st.session_state["calculation_method"])

    with control_column:
        render_controls(st.session_state["calculation_method"])
