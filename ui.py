"""Warstwa UI (Streamlit + Folium).

Ten moduł zawiera komponenty wizualne; logika i obliczenia znajdują się w `logic.py`.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium

from logic import (
    append_point,
    build_polyline_path,
    compute_center_of_gravity,
    compute_point_distances,
    ensure_points_dataframe,
    extract_marker_positions_from_drawings,
    get_map_center,
    points_dataframe_signature,
    read_points_from_uploaded_file,
    synchronize_points_dataframe_with_marker_positions,
)


def init_session_state() -> None:
    if "points_dataframe" not in st.session_state:
        st.session_state["points_dataframe"] = pd.DataFrame(columns=["longitude", "latitude", "transport_rate", "mass"])

    if "map_default_transport_rate" not in st.session_state:
        st.session_state["map_default_transport_rate"] = 1.0

    if "map_default_mass" not in st.session_state:
        st.session_state["map_default_mass"] = 1.0

    if "map_center_latitude" not in st.session_state:
        st.session_state["map_center_latitude"] = 52.2297

    if "map_center_longitude" not in st.session_state:
        st.session_state["map_center_longitude"] = 21.0122

    if "map_zoom_level" not in st.session_state:
        st.session_state["map_zoom_level"] = 11

    if "map_marker_positions_snapshot" not in st.session_state:
        st.session_state["map_marker_positions_snapshot"] = ()


def render_controls() -> None:
    st.subheader("Dane wejściowe")

    uploaded_file = st.file_uploader(
        "Wczytaj punkty z pliku (CSV/XLSX). Kolumny: longitude/latitude oraz opcjonalnie transport_rate/mass",
        type=["csv", "xlsx", "xls"],
    )
    if uploaded_file is not None:
        uploaded_points_dataframe = read_points_from_uploaded_file(uploaded_file)
        if len(uploaded_points_dataframe) > 0:
            st.session_state["points_dataframe"] = uploaded_points_dataframe
            st.session_state["map_marker_positions_snapshot"] = ()

    points_dataframe = ensure_points_dataframe(st.session_state["points_dataframe"])

    data_editor_fn = getattr(st, "data_editor", None)
    if data_editor_fn is not None:
        edited = data_editor_fn(points_dataframe, num_rows="dynamic", width="stretch")
        st.session_state["points_dataframe"] = ensure_points_dataframe(edited)
    else:
        st.dataframe(points_dataframe, width="stretch")

    st.divider()
    st.subheader("Dodaj punkt ręcznie")

    manual_longitude = st.number_input("X (longitude)", value=21.012200, format="%.6f")
    manual_latitude = st.number_input("Y (latitude)", value=52.229700, format="%.6f")
    manual_transport_rate = st.number_input("ST (stawka transportowa)", value=1.0, min_value=0.0, format="%.6f")
    manual_mass = st.number_input("M (masa)", value=1.0, min_value=0.0, format="%.6f")

    if st.button("Dodaj punkt", width="stretch"):
        st.session_state["points_dataframe"] = append_point(
            st.session_state["points_dataframe"],
            manual_longitude,
            manual_latitude,
            manual_transport_rate,
            manual_mass,
        )
        st.session_state["map_marker_positions_snapshot"] = ()

    if st.button("Wyczyść wszystkie punkty", width="stretch"):
        st.session_state["points_dataframe"] = pd.DataFrame(columns=["longitude", "latitude", "transport_rate", "mass"])
        st.session_state["map_marker_positions_snapshot"] = ()

    points_dataframe = ensure_points_dataframe(st.session_state["points_dataframe"])
    centroid_longitude, centroid_latitude, weighted_distance_sum = compute_center_of_gravity(points_dataframe)

    st.divider()
    st.subheader("Wynik")

    if len(points_dataframe) == 0:
        st.info("Dodaj co najmniej jeden punkt, aby wyliczyć X i Y.")
    else:
        st.metric("X (longitude)", f"{centroid_longitude:.6f}")
        st.metric("Y (latitude)", f"{centroid_latitude:.6f}")
        st.metric("Suma ważona odległości euklidesowych", f"{weighted_distance_sum:.6f}")

        distances_df = compute_point_distances(points_dataframe, centroid_longitude, centroid_latitude)
        st.subheader("Odległości (metryka euklidesowa)")
        st.dataframe(distances_df, width="stretch")

    st.divider()
    st.subheader("Domyślne wartości dla punktów dodanych z mapy")
    st.session_state["map_default_transport_rate"] = st.number_input(
        "Domyślne ST",
        value=float(st.session_state["map_default_transport_rate"]),
        min_value=0.0,
        format="%.6f",
    )
    st.session_state["map_default_mass"] = st.number_input(
        "Domyślne M",
        value=float(st.session_state["map_default_mass"]),
        min_value=0.0,
        format="%.6f",
    )
    st.write(
        "Dodawanie: wybierz narzędzie markera na mapie i kliknij w miejsce punktu. "
        "Przesuwanie/usuwanie: użyj trybu edycji/usuwania w panelu narzędzi mapy."
    )


def render_map() -> None:
    points_dataframe = ensure_points_dataframe(st.session_state["points_dataframe"])
    centroid_longitude, centroid_latitude, _ = compute_center_of_gravity(points_dataframe)

    map_center_longitude, map_center_latitude = get_map_center(points_dataframe, centroid_longitude, centroid_latitude)

    if "map_center_latitude" not in st.session_state or "map_center_longitude" not in st.session_state:
        st.session_state["map_center_latitude"] = float(map_center_latitude)
        st.session_state["map_center_longitude"] = float(map_center_longitude)

    if "map_zoom_level" not in st.session_state:
        st.session_state["map_zoom_level"] = 11

    polyline_path = build_polyline_path(points_dataframe)

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

    if len(polyline_path) >= 4:
        folium.PolyLine(
            locations=[[lat, lon] for lon, lat in polyline_path],
            weight=3,
        ).add_to(folium_map)

    if len(points_dataframe) > 0:
        folium.CircleMarker(
            location=[float(centroid_latitude), float(centroid_longitude)],
            radius=14,
            color="red",
            fill=True,
            fill_color="red",
            fill_opacity=0.95,
        ).add_to(folium_map)

    map_interaction = st_folium(
        folium_map,
        height=600,
        use_container_width=True,
        key="interactive_folium_map",
        returned_objects=["all_drawings", "last_active_drawing", "center", "zoom"],
        center=[float(st.session_state["map_center_latitude"]), float(st.session_state["map_center_longitude"])],
        zoom=int(st.session_state["map_zoom_level"]),
    )

    if not isinstance(map_interaction, dict):
        return

    returned_center = map_interaction.get("center")
    returned_zoom = map_interaction.get("zoom")

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

    last_active_drawing = map_interaction.get("last_active_drawing")
    marker_positions = extract_marker_positions_from_drawings(map_interaction.get("all_drawings"))
    marker_positions = sorted(marker_positions, key=lambda item: (float(item[0]), float(item[1])))

    points_positions_signature = tuple(
        sorted(
            (
                (round(float(row["longitude"]), 8), round(float(row["latitude"]), 8))
                for _, row in ensure_points_dataframe(st.session_state["points_dataframe"]).iterrows()
            ),
            key=lambda item: (float(item[0]), float(item[1])),
        )
    )

    if last_active_drawing is None and len(marker_positions) == 0:
        if points_positions_signature != tuple(st.session_state.get("map_marker_positions_snapshot", ())):
            st.session_state["map_marker_positions_snapshot"] = points_positions_signature
        return

    marker_positions_signature = tuple((round(float(lon), 8), round(float(lat), 8)) for lon, lat in marker_positions)

    if marker_positions_signature != tuple(st.session_state.get("map_marker_positions_snapshot", ())):
        previous_df = ensure_points_dataframe(st.session_state["points_dataframe"])
        previous_sig = points_dataframe_signature(previous_df)

        synced_df = synchronize_points_dataframe_with_marker_positions(
            previous_df,
            marker_positions,
            float(st.session_state["map_default_transport_rate"]),
            float(st.session_state["map_default_mass"]),
        )

        new_sig = points_dataframe_signature(synced_df)
        st.session_state["points_dataframe"] = synced_df

        updated_points_positions_signature = tuple(
            sorted(
                (
                    (round(float(row["longitude"]), 8), round(float(row["latitude"]), 8))
                    for _, row in ensure_points_dataframe(st.session_state["points_dataframe"]).iterrows()
                ),
                key=lambda item: (float(item[0]), float(item[1])),
            )
        )
        st.session_state["map_marker_positions_snapshot"] = updated_points_positions_signature

        if new_sig != previous_sig:
            st.rerun()


def run_app() -> None:
    st.set_page_config(page_title="Środek ciężkości logistyczny", layout="wide")

    init_session_state()

    st.title("Lokalizacja punktu logistycznego metodą środka ciężkości")
    st.write(
        "Współrzędne wyznaczane są wzorami: X = Σ(ST·M·X) / Σ(ST·M) oraz "
        "Y = Σ(ST·M·Y) / Σ(ST·M). Odległości liczone są metryką euklidesową: "
        "d = √((X−Xn)² + (Y−Yn)²)."
    )

    control_column, map_column = st.columns([1, 2], gap="large")

    with control_column:
        render_controls()

    with map_column:
        render_map()
