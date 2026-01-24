from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

from logic import (
    append_point,
    compute_center_of_gravity_details,
    compute_topsis_details,
    ensure_points_dataframe,
    get_map_center,
    get_topsis_candidate_criteria_columns,
    points_dataframe_signature,
    read_points_from_uploaded_file_with_status,
)


def init_session_state() -> None:
    if "points_dataframe" not in st.session_state:
        st.session_state["points_dataframe"] = pd.DataFrame(columns=["longitude", "latitude"])

    if "active_page" not in st.session_state:
        st.session_state["active_page"] = "Obliczenia"

    if "calculation_method" not in st.session_state:
        st.session_state["calculation_method"] = "Środek ciężkości"

    if "guided_mode" not in st.session_state:
        st.session_state["guided_mode"] = True

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

    if "interactive_folium_map_key_version" not in st.session_state:
        st.session_state["interactive_folium_map_key_version"] = 0

    if "map_last_clicked_latitude" not in st.session_state:
        st.session_state["map_last_clicked_latitude"] = None

    if "map_last_clicked_longitude" not in st.session_state:
        st.session_state["map_last_clicked_longitude"] = None


def get_interactive_folium_map_key() -> str:
    return f"interactive_folium_map_{int(st.session_state['interactive_folium_map_key_version'])}"


def bump_interactive_folium_map_key() -> None:
    st.session_state["interactive_folium_map_key_version"] = int(st.session_state["interactive_folium_map_key_version"]) + 1


def render_sidebar_menu() -> Tuple[str, str, bool]:
    with st.sidebar:
        st.title("Nawigacja")

        active_page = st.radio(
            "Widok",
            options=["Obliczenia", "Wyjaśnienie metod", "Pomoc"],
            index=["Obliczenia", "Wyjaśnienie metod", "Pomoc"].index(str(st.session_state.get("active_page", "Obliczenia"))),
            help="Wybierz, co chcesz teraz zrobić: obliczenia, opis metod albo pomoc.",
            key="active_page_radio",
        )
        st.session_state["active_page"] = str(active_page)

        calculation_method = st.radio(
            "Metoda obliczeń",
            options=["Środek ciężkości", "TOPSIS"],
            index=0 if str(st.session_state.get("calculation_method", "Środek ciężkości")) != "TOPSIS" else 1,
            help="Zmień sposób wyznaczania wyniku. Środek ciężkości liczy punkt jako średnią ważoną. TOPSIS tworzy ranking punktów na podstawie kryteriów.",
            key="calculation_method_radio",
        )

        if str(calculation_method) != str(st.session_state.get("calculation_method")):
            st.session_state["calculation_method"] = str(calculation_method)
            bump_interactive_folium_map_key()

        guided_mode = st.toggle(
            "Tryb prowadzenia krok po kroku",
            value=bool(st.session_state.get("guided_mode", True)),
            help="Gdy włączone, aplikacja wyjaśnia każdy etap i pokazuje, skąd biorą się wyniki oraz wzory.",
            key="guided_mode_toggle",
        )
        st.session_state["guided_mode"] = bool(guided_mode)

        st.divider()
        render_context_help_in_sidebar(str(st.session_state.get("active_page")), str(st.session_state.get("calculation_method")), bool(st.session_state.get("guided_mode")))

    return str(st.session_state.get("active_page")), str(st.session_state.get("calculation_method")), bool(st.session_state.get("guided_mode"))


def render_context_help_in_sidebar(active_page: str, calculation_method: str, guided_mode: bool) -> None:
    st.subheader("Pomoc kontekstowa")

    if active_page == "Obliczenia":
        st.write("Mapa: kliknij, aby wskazać współrzędne. Potem dodaj punkt przyciskiem w sekcji danych.")
        if calculation_method == "Środek ciężkości":
            st.write("Wymagane: współrzędne (X, Y). Zalecane: stawka transportowa i masa, bo tworzą wagę punktu.")
        else:
            st.write("Wymagane: współrzędne (X, Y) oraz co najmniej jedno kryterium liczbowe dla TOPSIS.")
            st.write("Dla każdego kryterium ustaw wagę i typ: korzyść (więcej lepiej) albo koszt (mniej lepiej).")
        if guided_mode:
            st.write("Tryb prowadzenia: aplikacja pokaże kroki obliczeń i tabele pośrednie.")
    elif active_page == "Wyjaśnienie metod":
        st.write("Znajdziesz tu opis metody środka ciężkości i TOPSIS, wraz z interpretacją wyników.")
    else:
        st.write("Jeśli coś nie działa, zacznij od sekcji „Najczęstsze problemy”.")


def render_methods_page() -> None:
    st.header("Wyjaśnienie metod")

    st.subheader("Metoda środka ciężkości")
    st.write("Cel: znaleźć punkt, który jest średnią ważoną punktów wejściowych.")
    st.write("Waga punktu to iloczyn: stawka transportowa × masa.")
    st.latex(r"w_i = ST_i \cdot M_i")
    st.latex(r"X = \frac{\sum (w_i \cdot X_i)}{\sum w_i} \qquad Y = \frac{\sum (w_i \cdot Y_i)}{\sum w_i}")
    st.write("Interpretacja: jeśli waga jest większa, punkt mocniej „ciągnie” wynik w swoją stronę.")
    st.write("Dodatkowo pokazujemy sumę ważonych odległości euklidesowych, aby zobaczyć łączny „koszt odległości” w tym uproszczonym modelu.")
    st.latex(r"d_i = \sqrt{(X - X_i)^2 + (Y - Y_i)^2} \qquad \sum (w_i \cdot d_i)")

    st.divider()

    st.subheader("TOPSIS")
    st.write("Cel: ułożyć ranking punktów na podstawie wielu kryteriów.")
    st.write("Każdy punkt to alternatywa, a kolumny kryteriów to liczby opisujące tę alternatywę (np. koszt, czas, ryzyko, dostępność).")
    st.write("Najpierw kryteria są normalizowane, potem uwzględniane są wagi, a następnie liczymy odległości do ideału najlepszego i najgorszego.")
    st.latex(r"s_i = \frac{d^-_i}{d^+_i + d^-_i}")
    st.write("Interpretacja: im większy wynik, tym punkt bliżej ideału najlepszego i dalej od ideału najgorszego.")


def render_help_page() -> None:
    st.header("Pomoc")

    st.subheader("Najczęstsze problemy")
    st.write("1) Brak wyniku: upewnij się, że masz co najmniej jeden punkt, a dla TOPSIS co najmniej jedno kryterium.")
    st.write("2) Plik wczytuje się, ale nie ma punktów: sprawdź, czy w pliku są kolumny z długością i szerokością geograficzną.")
    st.write("3) Punkt z mapy nie dodaje się: kliknij w mapę, potem użyj przycisku „Dodaj punkt z mapy” w sekcji danych.")

    st.divider()

    st.subheader("Wymagania danych")
    st.write("Współrzędne: długość geograficzna (X) i szerokość geograficzna (Y).")
    st.write("Środek ciężkości: możesz dodać stawkę transportową i masę. Jeśli ich nie podasz, przyjmujemy wartości 1.")
    st.write("TOPSIS: dodaj własne kryteria liczbowe. Dla każdego ustaw wagę oraz typ (korzyść/koszt).")

    st.divider()

    st.subheader("Czytelna interpretacja")
    st.write("Jeśli włączysz tryb prowadzenia, zobaczysz dokładne kroki obliczeń, tabele pośrednie i wyjaśnienie wzorów.")


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


def get_points_table_column_config_for_centroid() -> Dict[str, object]:
    return {
        "longitude": st.column_config.NumberColumn(
            "Długość geograficzna (X)",
            help="Współrzędna X. Typowo zakres: od -180 do 180.",
            format="%.6f",
        ),
        "latitude": st.column_config.NumberColumn(
            "Szerokość geograficzna (Y)",
            help="Współrzędna Y. Typowo zakres: od -90 do 90.",
            format="%.6f",
        ),
        "transport_rate": st.column_config.NumberColumn(
            "Stawka transportowa",
            help="Wartość, która wraz z masą tworzy wagę punktu. Waga punktu = stawka transportowa × masa.",
            format="%.6f",
        ),
        "mass": st.column_config.NumberColumn(
            "Masa",
            help="Wartość, która wraz ze stawką transportową tworzy wagę punktu. Waga punktu = stawka transportowa × masa.",
            format="%.6f",
        ),
    }


def get_points_table_column_config_for_topsis(points_dataframe: pd.DataFrame, selected_criteria_columns: List[str]) -> Dict[str, object]:
    column_config: Dict[str, object] = {
        "longitude": st.column_config.NumberColumn(
            "Długość geograficzna (X)",
            help="Współrzędna X. Typowo zakres: od -180 do 180.",
            format="%.6f",
        ),
        "latitude": st.column_config.NumberColumn(
            "Szerokość geograficzna (Y)",
            help="Współrzędna Y. Typowo zakres: od -90 do 90.",
            format="%.6f",
        ),
    }

    for column_name in points_dataframe.columns:
        normalized_column_name = str(column_name).strip().lower()
        if normalized_column_name in ("longitude", "latitude"):
            continue
        if normalized_column_name in selected_criteria_columns:
            column_config[normalized_column_name] = st.column_config.NumberColumn(
                f"Kryterium: {normalized_column_name}",
                help="Wartość liczbowa kryterium dla tego punktu. Upewnij się, że wszystkie punkty mają sensowne wartości.",
                format="%.6f",
            )
        else:
            column_config[normalized_column_name] = st.column_config.NumberColumn(
                f"Kolumna: {normalized_column_name}",
                help="Kolumna liczbowa, której nie używasz w rankingu, dopóki nie zaznaczysz jej jako kryterium.",
                format="%.6f",
            )

    return column_config


def render_onboarding(calculation_method: str, guided_mode: bool) -> None:
    if not guided_mode:
        return

    points_dataframe = ensure_points_dataframe(st.session_state["points_dataframe"], include_transport_rate_and_mass=False)

    step_has_points = len(points_dataframe) > 0
    step_has_topsis_criteria = True
    if calculation_method == "TOPSIS":
        available_criteria_columns = get_topsis_candidate_criteria_columns(points_dataframe)
        update_topsis_state_for_available_criteria(available_criteria_columns)
        selected_criteria_columns = list(st.session_state.get("topsis_selected_criteria_columns", []))
        step_has_topsis_criteria = len(selected_criteria_columns) > 0

    completed_steps = 0
    if step_has_points:
        completed_steps += 1
    if calculation_method == "TOPSIS" and step_has_topsis_criteria:
        completed_steps += 1
    if calculation_method != "TOPSIS" and step_has_points:
        completed_steps += 1

    step_count = 2 if calculation_method == "TOPSIS" else 2
    progress_value = float(completed_steps) / float(step_count) if step_count > 0 else 0.0

    st.subheader("Przewodnik: co teraz zrobić")
    st.progress(progress_value)
    if not step_has_points:
        st.info("Krok 1: dodaj punkty. Możesz je wczytać z pliku, dodać ręcznie albo kliknąć na mapie i dodać przyciskiem.")
        return

    if calculation_method == "TOPSIS" and not step_has_topsis_criteria:
        st.info("Krok 2: wybierz kryteria TOPSIS. Zaznacz przynajmniej jedno kryterium w sekcji konfiguracji.")
        return

    st.success("Masz komplet minimum danych. Przejdź do sekcji „Wynik”, aby zobaczyć rezultat i wyjaśnienie obliczeń.")


def render_file_import_section(calculation_method: str) -> None:
    st.subheader("Wczytanie danych z pliku")

    uploaded_file = st.file_uploader(
        "Wybierz plik z punktami (CSV lub Excel).",
        type=["csv", "xlsx", "xls"],
        help="W pliku muszą być kolumny ze współrzędnymi. Aplikacja rozpoznaje nazwy: longitude/latitude, lon/lat albo x/y.",
    )

    if uploaded_file is None:
        st.caption("Jeśli nie masz pliku, możesz dodać punkty ręcznie lub z mapy.")
        return

    uploaded_points_dataframe, message_text, message_level = read_points_from_uploaded_file_with_status(uploaded_file)
    if message_level == "success":
        st.success(message_text)
    elif message_level == "warning":
        st.warning(message_text)
    elif message_level == "error":
        st.error(message_text)
    else:
        st.info(message_text)

    if len(uploaded_points_dataframe) > 0:
        st.session_state["points_dataframe"] = uploaded_points_dataframe
        bump_interactive_folium_map_key()


def render_add_point_from_map_section(calculation_method: str) -> None:
    st.subheader("Dodanie punktu na podstawie kliknięcia na mapie")

    last_clicked_longitude = st.session_state.get("map_last_clicked_longitude")
    last_clicked_latitude = st.session_state.get("map_last_clicked_latitude")

    if last_clicked_longitude is None or last_clicked_latitude is None:
        st.info("Kliknij na mapie po prawej stronie, aby wybrać miejsce. Potem wróć tutaj i dodaj punkt.")
        return

    st.write(f"Wybrane współrzędne z mapy: X={float(last_clicked_longitude):.6f}, Y={float(last_clicked_latitude):.6f}")

    points_dataframe = ensure_points_dataframe(st.session_state["points_dataframe"], include_transport_rate_and_mass=False)

    if calculation_method == "TOPSIS":
        available_criteria_columns = get_topsis_candidate_criteria_columns(points_dataframe)
        update_topsis_state_for_available_criteria(available_criteria_columns)
        selected_criteria_columns = list(st.session_state.get("topsis_selected_criteria_columns", []))
        default_values_by_criteria: Dict[str, float] = dict(st.session_state.get("topsis_default_values_by_criteria", {}))

        additional_columns_values: Dict[str, float] = {}
        for criterion_name in selected_criteria_columns:
            additional_columns_values[str(criterion_name).strip().lower()] = float(default_values_by_criteria.get(str(criterion_name).strip().lower(), 1.0))

        if st.button(
            "Dodaj punkt z mapy do tabeli",
            use_container_width=True,
            help="Doda punkt do tabeli z domyślnymi wartościami kryteriów TOPSIS.",
            key="add_point_from_map_topsis_button",
        ):
            st.session_state["points_dataframe"] = append_point(
                st.session_state["points_dataframe"],
                float(last_clicked_longitude),
                float(last_clicked_latitude),
                transport_rate=None,
                mass=None,
                additional_columns_values=additional_columns_values,
            )
            st.success("Dodano punkt z mapy.")
            bump_interactive_folium_map_key()
        return

    default_transport_rate = float(st.session_state.get("map_default_transport_rate", 1.0))
    default_mass = float(st.session_state.get("map_default_mass", 1.0))

    if st.button(
        "Dodaj punkt z mapy do tabeli",
        use_container_width=True,
        help="Doda punkt do tabeli z domyślną stawką transportową i masą.",
        key="add_point_from_map_centroid_button",
    ):
        st.session_state["points_dataframe"] = append_point(
            st.session_state["points_dataframe"],
            float(last_clicked_longitude),
            float(last_clicked_latitude),
            transport_rate=float(default_transport_rate),
            mass=float(default_mass),
            additional_columns_values=None,
        )
        st.success("Dodano punkt z mapy.")
        bump_interactive_folium_map_key()


def render_centroid_controls(guided_mode: bool) -> None:
    st.header("Dane i obliczenia: metoda środka ciężkości")

    if guided_mode:
        with st.expander("Co oznacza ta metoda i jakie dane są potrzebne?", expanded=False):
            st.write("Ta metoda wyznacza wynik jako średnią ważoną współrzędnych.")
            st.write("Każdy punkt ma wagę: stawka transportowa × masa. Im większa waga, tym mocniejszy wpływ punktu na wynik.")
            st.latex(r"w_i = ST_i \cdot M_i")
            st.latex(r"X = \frac{\sum (w_i \cdot X_i)}{\sum w_i} \qquad Y = \frac{\sum (w_i \cdot Y_i)}{\sum w_i}")

    render_file_import_section("Środek ciężkości")

    points_dataframe = ensure_points_dataframe(st.session_state["points_dataframe"], include_transport_rate_and_mass=True)

    st.subheader("Tabela punktów")
    st.caption("Możesz edytować wartości w tabeli. Jeśli dopiszesz nowe wiersze, będą one traktowane jako nowe punkty.")

    data_editor_function = getattr(st, "data_editor", None)
    display_dataframe = points_dataframe.copy()
    display_dataframe = display_dataframe[["longitude", "latitude", "transport_rate", "mass"]]

    if data_editor_function is not None:
        previous_signature = points_dataframe_signature(display_dataframe)
        edited_display_dataframe = data_editor_function(
            display_dataframe,
            num_rows="dynamic",
            use_container_width=True,
            column_config=get_points_table_column_config_for_centroid(),
        )
        edited_display_dataframe = ensure_points_dataframe(edited_display_dataframe, include_transport_rate_and_mass=True)
        edited_signature = points_dataframe_signature(edited_display_dataframe)
        if edited_signature != previous_signature:
            full_points_dataframe = ensure_points_dataframe(st.session_state["points_dataframe"], include_transport_rate_and_mass=True)
            for column_name in ["longitude", "latitude", "transport_rate", "mass"]:
                full_points_dataframe[column_name] = edited_display_dataframe[column_name]
            st.session_state["points_dataframe"] = full_points_dataframe
            st.success("Zapisano zmiany w tabeli.")
            bump_interactive_folium_map_key()
    else:
        st.dataframe(display_dataframe, use_container_width=True)

    st.divider()
    render_add_point_from_map_section("Środek ciężkości")

    st.divider()
    st.subheader("Dodaj punkt ręcznie")

    with st.form("centroid_add_point_form", clear_on_submit=False):
        manual_longitude = st.number_input(
            "Długość geograficzna (X)",
            value=21.012200,
            format="%.6f",
            help="Podaj długość geograficzną punktu. Przykład dla Warszawy to około 21.0122.",
            key="centroid_manual_longitude",
        )
        manual_latitude = st.number_input(
            "Szerokość geograficzna (Y)",
            value=52.229700,
            format="%.6f",
            help="Podaj szerokość geograficzną punktu. Przykład dla Warszawy to około 52.2297.",
            key="centroid_manual_latitude",
        )
        manual_transport_rate = st.number_input(
            "Stawka transportowa",
            value=1.0,
            min_value=0.0,
            format="%.6f",
            help="Wartość wpływa na wagę punktu. Waga = stawka transportowa × masa.",
            key="centroid_manual_transport_rate",
        )
        manual_mass = st.number_input(
            "Masa",
            value=1.0,
            min_value=0.0,
            format="%.6f",
            help="Wartość wpływa na wagę punktu. Waga = stawka transportowa × masa.",
            key="centroid_manual_mass",
        )
        submit_add_point = st.form_submit_button(
            "Dodaj punkt",
            use_container_width=True,
            help="Doda punkt do tabeli.",
        )

    if submit_add_point:
        st.session_state["points_dataframe"] = append_point(
            st.session_state["points_dataframe"],
            manual_longitude,
            manual_latitude,
            transport_rate=manual_transport_rate,
            mass=manual_mass,
            additional_columns_values=None,
        )
        st.success("Dodano punkt.")
        bump_interactive_folium_map_key()

    st.divider()
    st.subheader("Domyślne wartości dla punktu dodawanego z mapy")

    st.session_state["map_default_transport_rate"] = st.number_input(
        "Domyślna stawka transportowa",
        value=float(st.session_state.get("map_default_transport_rate", 1.0)),
        min_value=0.0,
        format="%.6f",
        help="Ta wartość zostanie użyta, gdy dodasz punkt z mapy przyciskiem, a nie ustawisz stawki ręcznie.",
        key="centroid_map_default_transport_rate",
    )
    st.session_state["map_default_mass"] = st.number_input(
        "Domyślna masa",
        value=float(st.session_state.get("map_default_mass", 1.0)),
        min_value=0.0,
        format="%.6f",
        help="Ta wartość zostanie użyta, gdy dodasz punkt z mapy przyciskiem, a nie ustawisz masy ręcznie.",
        key="centroid_map_default_mass",
    )

    st.divider()
    st.subheader("Wynik i wyjaśnienie")

    updated_points_dataframe = ensure_points_dataframe(st.session_state["points_dataframe"], include_transport_rate_and_mass=True)

    if len(updated_points_dataframe) == 0:
        st.info("Dodaj co najmniej jeden punkt, aby policzyć wynik.")
        return

    center_of_gravity_details = compute_center_of_gravity_details(updated_points_dataframe)

    st.metric("Długość geograficzna wyniku (X)", f"{float(center_of_gravity_details.centroid_longitude):.6f}")
    st.metric("Szerokość geograficzna wyniku (Y)", f"{float(center_of_gravity_details.centroid_latitude):.6f}")
    st.metric("Suma ważonych odległości euklidesowych", f"{float(center_of_gravity_details.weighted_distance_sum):.6f}")

    if guided_mode:
        with st.expander("Skąd się wziął wynik? Pokaż krok po kroku", expanded=True):
            st.write("1) Dla każdego punktu liczymy wagę:")
            st.latex(r"w_i = ST_i \cdot M_i")
            st.write("2) Liczymy sumy ważone współrzędnych:")
            st.latex(r"\sum (w_i \cdot X_i) \quad\text{oraz}\quad \sum (w_i \cdot Y_i)")
            st.write("3) Dzielimy przez sumę wag, aby uzyskać średnią ważoną:")
            st.latex(r"X = \frac{\sum (w_i \cdot X_i)}{\sum w_i} \qquad Y = \frac{\sum (w_i \cdot Y_i)}{\sum w_i}")

            st.write(f"Suma wag: {float(center_of_gravity_details.total_weight):.6f}")
            st.write(f"Suma (waga × X): {float(center_of_gravity_details.weighted_longitude_sum):.6f}")
            st.write(f"Suma (waga × Y): {float(center_of_gravity_details.weighted_latitude_sum):.6f}")

            if bool(center_of_gravity_details.used_fallback_average):
                st.warning("Suma wag wyszła bliska zeru, więc wynik policzono jako zwykłą średnią współrzędnych.")

            st.divider()
            st.write("4) Dodatkowo liczymy odległość euklidesową od wyniku do każdego punktu i ważymy ją wagą:")
            st.latex(r"d_i = \sqrt{(X - X_i)^2 + (Y - Y_i)^2}")
            st.latex(r"\sum (w_i \cdot d_i)")
            st.write("Poniżej tabela pokazująca wagę i wkład każdego punktu:")

            breakdown_dataframe = center_of_gravity_details.per_point_breakdown_dataframe.copy()
            breakdown_dataframe = breakdown_dataframe.rename(
                columns={
                    "longitude": "Długość geograficzna (X)",
                    "latitude": "Szerokość geograficzna (Y)",
                    "transport_rate": "Stawka transportowa",
                    "mass": "Masa",
                    "point_weight": "Waga punktu",
                    "weighted_longitude": "Waga × X",
                    "weighted_latitude": "Waga × Y",
                    "euclidean_distance": "Odległość euklidesowa",
                    "weighted_euclidean_distance": "Waga × odległość",
                }
            )
            st.dataframe(breakdown_dataframe, use_container_width=True)

    if st.button(
        "Wyczyść wszystkie punkty",
        use_container_width=True,
        help="Usuwa wszystkie punkty i resetuje obliczenia.",
        key="centroid_clear_points_button",
    ):
        st.session_state["points_dataframe"] = pd.DataFrame(columns=["longitude", "latitude", "transport_rate", "mass"])
        st.session_state["map_last_clicked_latitude"] = None
        st.session_state["map_last_clicked_longitude"] = None
        st.success("Wyczyszczono dane.")
        bump_interactive_folium_map_key()


def render_topsis_controls(guided_mode: bool) -> None:
    st.header("Dane i obliczenia: TOPSIS")

    if guided_mode:
        with st.expander("Co oznacza TOPSIS i jakie dane są potrzebne?", expanded=False):
            st.write("TOPSIS tworzy ranking punktów na podstawie wielu kryteriów.")
            st.write("Każdy punkt ma współrzędne (X, Y) oraz wartości kryteriów (liczb).")
            st.write("Ustawiasz wagę kryterium (jak bardzo ma wpływać) oraz typ:")
            st.write("• Korzyść: większa wartość jest lepsza")
            st.write("• Koszt: mniejsza wartość jest lepsza")
            st.latex(r"s_i = \frac{d^-_i}{d^+_i + d^-_i}")

    render_file_import_section("TOPSIS")

    points_dataframe = ensure_points_dataframe(st.session_state["points_dataframe"], include_transport_rate_and_mass=False)

    st.subheader("Dodaj kryterium")
    st.caption("Kryterium to dodatkowa kolumna liczbowa opisująca punkt. Przykłady: koszt, czas, ryzyko, dostępność.")

    criterion_column_name = st.text_input(
        "Nazwa nowego kryterium",
        value="",
        help="Wpisz nazwę kolumny-kryterium. Używaj nazw prostych i bez spacji, np. koszt, czas, ryzyko.",
        key="topsis_criterion_column_name",
    )
    criterion_default_value = st.number_input(
        "Domyślna wartość nowego kryterium",
        value=1.0,
        format="%.6f",
        help="Wartość zostanie wpisana w istniejących punktach oraz będzie używana jako domyślna przy dodawaniu punktu z mapy.",
        key="topsis_criterion_default_value",
    )

    if st.button(
        "Dodaj kryterium do tabeli",
        use_container_width=True,
        help="Doda nową kolumnę do tabeli punktów.",
        key="topsis_add_criterion_button",
    ):
        normalized_criterion_column_name = str(criterion_column_name).strip().lower()
        if normalized_criterion_column_name:
            updated_points_dataframe = points_dataframe.copy()
            if normalized_criterion_column_name not in updated_points_dataframe.columns:
                updated_points_dataframe[normalized_criterion_column_name] = float(criterion_default_value)
                st.session_state["points_dataframe"] = updated_points_dataframe
                st.success("Dodano kryterium.")
                bump_interactive_folium_map_key()
            else:
                st.warning("Takie kryterium już istnieje.")
        else:
            st.warning("Podaj nazwę kryterium.")

    points_dataframe = ensure_points_dataframe(st.session_state["points_dataframe"], include_transport_rate_and_mass=False)

    available_criteria_columns = get_topsis_candidate_criteria_columns(points_dataframe)
    update_topsis_state_for_available_criteria(available_criteria_columns)

    st.divider()
    st.subheader("Wybór kryteriów do obliczeń")

    selected_criteria_columns = st.multiselect(
        "Które kryteria mają brać udział w rankingu?",
        options=available_criteria_columns,
        default=list(st.session_state.get("topsis_selected_criteria_columns", [])),
        help="Zaznacz kryteria, które mają wpływać na ranking. Musisz wybrać co najmniej jedno.",
        key="topsis_selected_criteria_columns_widget",
    )
    st.session_state["topsis_selected_criteria_columns"] = [str(column_name).strip().lower() for column_name in selected_criteria_columns if str(column_name).strip()]
    update_topsis_state_for_available_criteria(available_criteria_columns)

    selected_criteria_columns = list(st.session_state.get("topsis_selected_criteria_columns", []))
    criteria_weights: Dict[str, float] = dict(st.session_state.get("topsis_criteria_weights", {}))
    criteria_impacts: Dict[str, str] = dict(st.session_state.get("topsis_criteria_impacts", {}))
    default_values_by_criteria: Dict[str, float] = dict(st.session_state.get("topsis_default_values_by_criteria", {}))

    st.divider()
    st.subheader("Konfiguracja kryteriów")

    if len(selected_criteria_columns) == 0:
        st.info("Wybierz co najmniej jedno kryterium, aby ustawić wagi i typy.")
    else:
        for criterion_name in selected_criteria_columns:
            current_weight_value = float(criteria_weights.get(criterion_name, 1.0))
            current_impact_value = str(criteria_impacts.get(criterion_name, "benefit")).strip().lower()
            current_default_value = float(default_values_by_criteria.get(criterion_name, 1.0))

            criteria_weights[criterion_name] = st.number_input(
                f"Waga kryterium: {criterion_name}",
                value=float(current_weight_value),
                min_value=0.0,
                format="%.6f",
                help="Waga mówi, jak ważne jest kryterium względem innych. Liczy się proporcja wag.",
                key=f"topsis_weight_{criterion_name}",
            )

            selected_impact_label = "Korzyść (większe = lepiej)" if current_impact_value != "cost" else "Koszt (mniejsze = lepiej)"
            impact_label = st.selectbox(
                f"Typ kryterium: {criterion_name}",
                options=["Korzyść (większe = lepiej)", "Koszt (mniejsze = lepiej)"],
                index=0 if selected_impact_label.startswith("Korzyść") else 1,
                help="Wybierz, czy większa wartość jest lepsza (korzyść), czy gorsza (koszt).",
                key=f"topsis_impact_{criterion_name}",
            )
            criteria_impacts[criterion_name] = "benefit" if str(impact_label).startswith("Korzyść") else "cost"

            default_values_by_criteria[criterion_name] = st.number_input(
                f"Domyślna wartość przy dodaniu z mapy: {criterion_name}",
                value=float(current_default_value),
                format="%.6f",
                help="Ta wartość zostanie użyta, gdy dodasz punkt z mapy przyciskiem, a potem dopiero uzupełnisz dane w tabeli.",
                key=f"topsis_default_{criterion_name}",
            )

    st.session_state["topsis_criteria_weights"] = {str(key).strip().lower(): float(value) for key, value in criteria_weights.items()}
    st.session_state["topsis_criteria_impacts"] = {str(key).strip().lower(): str(value).strip().lower() for key, value in criteria_impacts.items()}
    st.session_state["topsis_default_values_by_criteria"] = {str(key).strip().lower(): float(value) for key, value in default_values_by_criteria.items()}

    st.divider()
    st.subheader("Tabela punktów (alternatywy)")

    data_editor_function = getattr(st, "data_editor", None)
    if data_editor_function is not None:
        previous_signature = points_dataframe_signature(points_dataframe)
        edited_points_dataframe = data_editor_function(
            points_dataframe,
            num_rows="dynamic",
            use_container_width=True,
            column_config=get_points_table_column_config_for_topsis(points_dataframe, selected_criteria_columns),
        )
        edited_points_dataframe = ensure_points_dataframe(edited_points_dataframe, include_transport_rate_and_mass=False)
        edited_signature = points_dataframe_signature(edited_points_dataframe)
        if edited_signature != previous_signature:
            st.session_state["points_dataframe"] = edited_points_dataframe
            st.success("Zapisano zmiany w tabeli.")
            bump_interactive_folium_map_key()
            points_dataframe = edited_points_dataframe
    else:
        st.dataframe(points_dataframe, use_container_width=True)

    st.divider()
    render_add_point_from_map_section("TOPSIS")

    st.divider()
    st.subheader("Dodaj punkt ręcznie")

    with st.form("topsis_add_point_form", clear_on_submit=False):
        manual_longitude = st.number_input(
            "Długość geograficzna (X)",
            value=21.012200,
            format="%.6f",
            help="Podaj długość geograficzną punktu.",
            key="topsis_manual_longitude",
        )
        manual_latitude = st.number_input(
            "Szerokość geograficzna (Y)",
            value=52.229700,
            format="%.6f",
            help="Podaj szerokość geograficzną punktu.",
            key="topsis_manual_latitude",
        )

        manual_criteria_values: Dict[str, float] = {}
        for criterion_name in selected_criteria_columns:
            manual_criteria_values[criterion_name] = st.number_input(
                f"Wartość kryterium: {criterion_name}",
                value=float(st.session_state["topsis_default_values_by_criteria"].get(criterion_name, 1.0)),
                format="%.6f",
                help="Ustaw wartość kryterium dla dodawanego punktu.",
                key=f"topsis_manual_{criterion_name}",
            )

        submit_add_point = st.form_submit_button(
            "Dodaj punkt",
            use_container_width=True,
            help="Doda punkt do tabeli.",
        )

    if submit_add_point:
        st.session_state["points_dataframe"] = append_point(
            st.session_state["points_dataframe"],
            manual_longitude,
            manual_latitude,
            transport_rate=None,
            mass=None,
            additional_columns_values=manual_criteria_values,
        )
        st.success("Dodano punkt.")
        bump_interactive_folium_map_key()

    st.divider()
    st.subheader("Wynik i wyjaśnienie")

    points_dataframe = ensure_points_dataframe(st.session_state["points_dataframe"], include_transport_rate_and_mass=False)

    if len(points_dataframe) == 0:
        st.info("Dodaj co najmniej jeden punkt, aby policzyć ranking.")
        return
    if len(selected_criteria_columns) == 0:
        st.info("Wybierz co najmniej jedno kryterium, aby policzyć ranking TOPSIS.")
        return

    topsis_details = compute_topsis_details(
        points_dataframe,
        criteria_columns=selected_criteria_columns,
        criteria_weights_by_name=st.session_state.get("topsis_criteria_weights", {}),
        criteria_impacts_by_name=st.session_state.get("topsis_criteria_impacts", {}),
    )

    if len(topsis_details.ranking_dataframe) == 0:
        st.info("Brak danych do obliczeń.")
        return

    best_row = topsis_details.ranking_dataframe.iloc[0]
    st.metric("Najlepszy wynik TOPSIS", f"{float(best_row.get('topsis_score', 0.0)):.6f}")
    st.metric("Długość geograficzna najlepszego punktu (X)", f"{float(best_row.get('longitude', 0.0)):.6f}")
    st.metric("Szerokość geograficzna najlepszego punktu (Y)", f"{float(best_row.get('latitude', 0.0)):.6f}")

    st.subheader("Ranking punktów")
    ranking_dataframe = topsis_details.ranking_dataframe.copy()
    ranking_dataframe = ranking_dataframe.rename(columns={"longitude": "Długość geograficzna (X)", "latitude": "Szerokość geograficzna (Y)", "topsis_score": "Wynik TOPSIS", "topsis_rank": "Pozycja w rankingu"})
    st.dataframe(ranking_dataframe, use_container_width=True)

    if guided_mode:
        with st.expander("Skąd się wziął ranking? Pokaż kroki obliczeń TOPSIS", expanded=True):
            st.write("Krok 1: macierz decyzyjna (same kryteria). Braki są traktowane jako 0.")
            st.dataframe(topsis_details.decision_matrix, use_container_width=True)

            st.divider()
            st.write("Krok 2: normalizacja wektorowa w kolumnach, aby różne skale nie dominowały.")
            st.latex(r"r_{ij} = \frac{x_{ij}}{\sqrt{\sum_i x_{ij}^2}}")
            st.dataframe(topsis_details.normalized_matrix, use_container_width=True)

            st.divider()
            st.write("Krok 3: wagi kryteriów (znormalizowane tak, aby sumowały się do 1).")
            weights_dataframe = pd.DataFrame(
                [{"kryterium": column_name, "waga_znormalizowana": float(weight_value)} for column_name, weight_value in topsis_details.normalized_weights_by_column.items()]
            )
            weights_dataframe = weights_dataframe.rename(columns={"kryterium": "Kryterium", "waga_znormalizowana": "Waga znormalizowana"})
            st.dataframe(weights_dataframe, use_container_width=True)

            st.divider()
            st.write("Krok 4: macierz znormalizowana i ważona.")
            st.latex(r"v_{ij} = r_{ij} \cdot w_j")
            st.dataframe(topsis_details.weighted_normalized_matrix, use_container_width=True)

            st.divider()
            st.write("Krok 5: ideał najlepszy i najgorszy dla każdego kryterium (korzyść/koszt).")
            ideal_rows: List[Dict[str, float]] = []
            for column_name in topsis_details.valid_criteria_columns:
                ideal_rows.append(
                    {
                        "kryterium": str(column_name),
                        "ideał_najlepszy": float(topsis_details.ideal_best_by_column.get(column_name, 0.0)),
                        "ideał_najgorszy": float(topsis_details.ideal_worst_by_column.get(column_name, 0.0)),
                        "typ": "korzyść" if str(st.session_state.get("topsis_criteria_impacts", {}).get(column_name, "benefit")).strip().lower() != "cost" else "koszt",
                    }
                )
            ideals_dataframe = pd.DataFrame(ideal_rows)
            ideals_dataframe = ideals_dataframe.rename(columns={"kryterium": "Kryterium", "ideał_najlepszy": "Ideał najlepszy", "ideał_najgorszy": "Ideał najgorszy", "typ": "Typ kryterium"})
            st.dataframe(ideals_dataframe, use_container_width=True)

            st.divider()
            st.write("Krok 6: odległości od ideału najlepszego i najgorszego oraz wynik końcowy.")
            st.latex(r"d^+_i = \sqrt{\sum_j (v_{ij} - v^+_j)^2} \qquad d^-_i = \sqrt{\sum_j (v_{ij} - v^-_j)^2}")
            st.latex(r"s_i = \frac{d^-_i}{d^+_i + d^-_i}")

            distances_dataframe = pd.DataFrame(
                {
                    "odległość_do_najlepszego": [float(value) for value in topsis_details.distances_to_best],
                    "odległość_do_najgorszego": [float(value) for value in topsis_details.distances_to_worst],
                    "wynik_topsis": [float(value) for value in topsis_details.topsis_scores],
                }
            )
            distances_dataframe = distances_dataframe.rename(
                columns={
                    "odległość_do_najlepszego": "Odległość do ideału najlepszego",
                    "odległość_do_najgorszego": "Odległość do ideału najgorszego",
                    "wynik_topsis": "Wynik TOPSIS",
                }
            )
            st.dataframe(distances_dataframe, use_container_width=True)

    if st.button(
        "Wyczyść wszystkie punkty",
        use_container_width=True,
        help="Usuwa wszystkie punkty i resetuje konfigurację rankingu.",
        key="topsis_clear_points_button",
    ):
        st.session_state["points_dataframe"] = pd.DataFrame(columns=["longitude", "latitude"])
        st.session_state["topsis_selected_criteria_columns"] = []
        st.session_state["topsis_criteria_weights"] = {}
        st.session_state["topsis_criteria_impacts"] = {}
        st.session_state["topsis_default_values_by_criteria"] = {}
        st.session_state["map_last_clicked_latitude"] = None
        st.session_state["map_last_clicked_longitude"] = None
        st.success("Wyczyszczono dane.")
        bump_interactive_folium_map_key()


def compute_result_marker_for_map(calculation_method: str, points_dataframe: pd.DataFrame) -> Tuple[float, float, str]:
    ensured_points_dataframe = ensure_points_dataframe(points_dataframe, include_transport_rate_and_mass=False)

    if calculation_method == "TOPSIS":
        available_criteria_columns = get_topsis_candidate_criteria_columns(ensured_points_dataframe)
        update_topsis_state_for_available_criteria(available_criteria_columns)
        selected_criteria_columns = list(st.session_state.get("topsis_selected_criteria_columns", []))
        if len(ensured_points_dataframe) == 0 or len(selected_criteria_columns) == 0:
            return 0.0, 0.0, ""
        topsis_details = compute_topsis_details(
            ensured_points_dataframe,
            criteria_columns=selected_criteria_columns,
            criteria_weights_by_name=st.session_state.get("topsis_criteria_weights", {}),
            criteria_impacts_by_name=st.session_state.get("topsis_criteria_impacts", {}),
        )
        if len(topsis_details.ranking_dataframe) == 0:
            return 0.0, 0.0, ""
        best_row = topsis_details.ranking_dataframe.iloc[0]
        best_longitude = float(best_row.get("longitude", 0.0))
        best_latitude = float(best_row.get("latitude", 0.0))
        best_score = float(best_row.get("topsis_score", 0.0))
        tooltip_text = f"Najlepszy punkt TOPSIS: wynik={best_score:.6f}, Y={best_latitude:.6f}, X={best_longitude:.6f}"
        return best_longitude, best_latitude, tooltip_text

    center_of_gravity_details = compute_center_of_gravity_details(ensure_points_dataframe(ensured_points_dataframe, include_transport_rate_and_mass=True))
    tooltip_text = f"Wynik środka ciężkości: Y={float(center_of_gravity_details.centroid_latitude):.6f}, X={float(center_of_gravity_details.centroid_longitude):.6f}"
    return float(center_of_gravity_details.centroid_longitude), float(center_of_gravity_details.centroid_latitude), tooltip_text


def render_map(calculation_method: str, guided_mode: bool) -> None:
    current_map_key = get_interactive_folium_map_key()
    previous_map_interaction = st.session_state.get(current_map_key)

    if isinstance(previous_map_interaction, dict):
        returned_center = previous_map_interaction.get("center")
        returned_zoom = previous_map_interaction.get("zoom")
        returned_last_clicked = previous_map_interaction.get("last_clicked")

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

        if isinstance(returned_last_clicked, dict) and "lat" in returned_last_clicked and "lng" in returned_last_clicked:
            try:
                st.session_state["map_last_clicked_latitude"] = float(returned_last_clicked["lat"])
                st.session_state["map_last_clicked_longitude"] = float(returned_last_clicked["lng"])
            except Exception:
                pass

    points_dataframe = ensure_points_dataframe(st.session_state["points_dataframe"], include_transport_rate_and_mass=False)

    result_longitude, result_latitude, result_tooltip_text = compute_result_marker_for_map(calculation_method, points_dataframe)

    map_center_longitude, map_center_latitude = get_map_center(points_dataframe, result_longitude, result_latitude)

    if "map_center_latitude" not in st.session_state or "map_center_longitude" not in st.session_state:
        st.session_state["map_center_latitude"] = float(map_center_latitude)
        st.session_state["map_center_longitude"] = float(map_center_longitude)

    if "map_zoom_level" not in st.session_state:
        st.session_state["map_zoom_level"] = 11

    st.subheader("Mapa")
    if guided_mode:
        st.caption("Kliknij na mapie, aby wskazać współrzędne nowego punktu. Następnie dodaj go przyciskiem w sekcji danych po lewej stronie.")
    else:
        st.caption("Kliknij na mapie, aby wskazać współrzędne nowego punktu.")

    folium_map = folium.Map(
        location=[float(st.session_state["map_center_latitude"]), float(st.session_state["map_center_longitude"])],
        zoom_start=int(st.session_state["map_zoom_level"]),
        control_scale=True,
        tiles=None,
    )

    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="© autorzy OpenStreetMap",
        name="Mapa bazowa",
        control=False,
    ).add_to(folium_map)

    for _, row in points_dataframe.iterrows():
        folium.Marker(
            location=[float(row["latitude"]), float(row["longitude"])],
            tooltip=f"Punkt: Y={float(row['latitude']):.6f}, X={float(row['longitude']):.6f}",
        ).add_to(folium_map)

    if len(points_dataframe) > 0 and (abs(float(result_longitude)) > 1e-12 or abs(float(result_latitude)) > 1e-12):
        folium.Marker(
            location=[float(result_latitude), float(result_longitude)],
            tooltip=result_tooltip_text,
            icon=folium.Icon(color="red"),
        ).add_to(folium_map)

    last_clicked_latitude = st.session_state.get("map_last_clicked_latitude")
    last_clicked_longitude = st.session_state.get("map_last_clicked_longitude")
    if last_clicked_latitude is not None and last_clicked_longitude is not None:
        folium.Marker(
            location=[float(last_clicked_latitude), float(last_clicked_longitude)],
            tooltip=f"Wybrane kliknięciem: Y={float(last_clicked_latitude):.6f}, X={float(last_clicked_longitude):.6f}",
            icon=folium.Icon(color="blue"),
        ).add_to(folium_map)

    st_folium(
        folium_map,
        height=600,
        use_container_width=True,
        key=current_map_key,
        returned_objects=["last_clicked", "center", "zoom"],
        center=[float(st.session_state["map_center_latitude"]), float(st.session_state["map_center_longitude"])],
        zoom=int(st.session_state["map_zoom_level"]),
    )


def run_app() -> None:
    st.set_page_config(page_title="Lokalizacja punktu logistycznego", layout="wide")

    init_session_state()

    active_page, calculation_method, guided_mode = render_sidebar_menu()

    st.title("Lokalizacja punktu logistycznego")
    st.caption("Dodaj punkty, wybierz metodę, a aplikacja policzy wynik i pokaże, skąd się wziął.")

    if active_page == "Wyjaśnienie metod":
        render_methods_page()
        return

    if active_page == "Pomoc":
        render_help_page()
        return

    render_onboarding(calculation_method, guided_mode)

    control_column, map_column = st.columns([1, 2], gap="large")

    with map_column:
        render_map(calculation_method, guided_mode)

    with control_column:
        if calculation_method == "TOPSIS":
            render_topsis_controls(guided_mode)
        else:
            render_centroid_controls(guided_mode)
