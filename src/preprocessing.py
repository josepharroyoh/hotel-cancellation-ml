"""Preparação dos dados de reservas.

Este arquivo guarda, em código, as decisões de limpeza do notebook 03 e as
variáveis de histórico do notebook 06. Os notebooks importam estas funções para
não repetir a limpeza toda vez e para usar sempre o mesmo tratamento.
"""

import numpy as np
import pandas as pd

DATA_URL = (
    "https://raw.githubusercontent.com/josepharroyoh/hotel-cancellation-ml/main/data/raw/hotel_booking.csv"
)

ALVO = "is_canceled"

# colunas de identificação artificiais (versão do Kaggle)
COLS_IDENTIFICACAO = ["name", "email", "phone-number", "credit_card"]

# colunas que revelam o resultado final da reserva (data leakage)
COLS_VAZAMENTO = ["reservation_status", "reservation_status_date"]

# colunas que só existem depois que a reserva foi feita
COLS_POS_RESERVA = ["assigned_room_type", "booking_changes", "days_in_waiting_list"]

MESES = ["January", "February", "March", "April", "May", "June", "July",
         "August", "September", "October", "November", "December"]

# variáveis usadas nos modelos (definidas no notebook 03)
VARIAVEIS_NUMERICAS = [
    "lead_time", "arrival_date_week_number", "arrival_date_day_of_month",
    "stays_in_weekend_nights", "stays_in_week_nights",
    "adults", "children", "babies",
    "is_repeated_guest", "previous_cancellations", "previous_bookings_not_canceled",
    "required_car_parking_spaces", "total_of_special_requests",
    "adr", "total_nights", "total_guests",
    "has_previous_cancellation", "zero_nights", "has_company", "has_agent",
]

VARIAVEIS_CATEGORICAS = [
    "hotel", "arrival_date_month", "meal", "country",
    "market_segment", "distribution_channel",
    "reserved_room_type", "deposit_type", "customer_type", "agent",
    "arrival_dow", "booking_month", "booking_dow",
]

VARIAVEIS = VARIAVEIS_NUMERICAS + VARIAVEIS_CATEGORICAS

# variáveis de histórico (notebook 06), calculadas só com reservas anteriores
VARIAVEIS_HISTORICO = ["hist_log_n_agent", "hist_taxa_agent",
                       "hist_log_n_country", "hist_taxa_country"]


def carregar(url=DATA_URL):
    return pd.read_csv(url)


def limpar(df):
    """Tira colunas que não entram no modelo e trata valores estranhos."""
    df = df.copy()
    df = df.drop(columns=COLS_IDENTIFICACAO + COLS_VAZAMENTO + COLS_POS_RESERVA)

    # children tem 4 ausentes; 0 é o valor natural
    df["children"] = df["children"].fillna(0)

    # reconstrói a data de chegada e estima a data da reserva
    mes = {nome: i for i, nome in enumerate(MESES, start=1)}
    df["arrival_date"] = pd.to_datetime(dict(
        year=df["arrival_date_year"],
        month=df["arrival_date_month"].map(mes),
        day=df["arrival_date_day_of_month"],
    ))
    df["booking_date"] = df["arrival_date"] - pd.to_timedelta(df["lead_time"], unit="D")

    # adr: um valor negativo (erro) e um valor absurdo de 5400
    df = df[df["adr"] >= 0].copy()
    df["adr"] = df["adr"].clip(upper=1000)

    # reserva sem nenhum hóspede não é uma reserva de verdade
    hospedes = df["adults"] + df["children"] + df["babies"]
    df = df[hospedes > 0].copy()

    return df.reset_index(drop=True)


def criar_variaveis(df):
    """Cria variáveis novas a partir das colunas que já existem."""
    df = df.copy()

    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df["has_previous_cancellation"] = (df["previous_cancellations"] > 0).astype(int)
    df["zero_nights"] = (df["total_nights"] == 0).astype(int)
    df["has_company"] = df["company"].notna().astype(int)
    df["has_agent"] = df["agent"].notna().astype(int)

    df["arrival_dow"] = df["arrival_date"].dt.dayofweek.astype(str)
    df["booking_month"] = df["booking_date"].dt.month.astype(str)
    df["booking_dow"] = df["booking_date"].dt.dayofweek.astype(str)

    # agent e country têm muitos valores diferentes; os raros viram "outro"
    df["agent"] = df["agent"].astype("Int64").astype("string").fillna("nenhum")
    df["country"] = df["country"].fillna("outro")
    for col in ["country", "agent"]:
        freq = df[col].value_counts()
        comuns = freq[freq >= 200].index
        df[col] = df[col].where(df[col].isin(comuns), "outro").astype(str)

    df = df.drop(columns=["company"])
    return df


def adicionar_historico(df, m=20):
    """Histórico de cancelamento da agência e do país, usando só o passado.

    Para cada reserva, olho as reservas da mesma agência (e do mesmo país) que
    chegaram antes do dia em que a reserva foi feita. Para essas o resultado
    (cancelou ou não) já era conhecido naquele dia.

    Para cada chave crio duas variáveis:
    - hist_log_n: log(1 + número dessas reservas anteriores);
    - hist_taxa: taxa de cancelamento delas, puxada para a taxa geral do
      passado quando há poucas reservas (m funciona como m reservas
      "fictícias" com a taxa geral).
    Usa as colunas originais agent e country (antes do agrupamento).
    """
    df = df.copy()
    chaves = {
        "agent": df["agent"].astype("Int64").astype("string").fillna("nenhum").astype(str),
        "country": df["country"].fillna("desconhecido"),
    }
    consulta = pd.DataFrame({"linha": df.index, "data": df["booking_date"]})
    consulta = consulta.sort_values("data", kind="stable")

    # taxa geral de cancelamento das reservas que já tinham chegado em cada data
    ev = df[["arrival_date", ALVO]].sort_values("arrival_date", kind="stable")
    geral = pd.DataFrame({"data": ev["arrival_date"].values,
                          "n_geral": np.arange(1, len(ev) + 1),
                          "c_geral": ev[ALVO].cumsum().values})
    q = pd.merge_asof(consulta, geral, on="data", allow_exact_matches=False)
    taxa_geral = (q["c_geral"] / q["n_geral"]).values

    for nome, chave in chaves.items():
        ev = pd.DataFrame({"chave": chave, "data": df["arrival_date"], "y": df[ALVO]})
        ev = ev.sort_values("data", kind="stable")
        ev["n"] = ev.groupby("chave").cumcount() + 1
        ev["c"] = ev.groupby("chave")["y"].cumsum()
        c_q = consulta.assign(chave=chave.loc[consulta["linha"]].values)
        # allow_exact_matches=False: só entram chegadas de antes do dia da reserva
        r = pd.merge_asof(c_q, ev[["chave", "data", "n", "c"]], on="data", by="chave",
                          allow_exact_matches=False)
        n = r["n"].fillna(0).values
        c = r["c"].fillna(0).values
        df.loc[r["linha"].values, f"hist_log_n_{nome}"] = np.log1p(n)
        df.loc[r["linha"].values, f"hist_taxa_{nome}"] = (c + m * taxa_geral) / (n + m)
    return df


def preparar(url=DATA_URL, historico=False, m=20):
    """Base pronta para a modelagem (ainda sem imputação, encoding ou escala).

    Com historico=True inclui também as variáveis de histórico (notebook 06).
    """
    df = limpar(carregar(url))
    if historico:
        df = adicionar_historico(df, m=m)
    return criar_variaveis(df)


def separar_treino_teste(df, frac_treino=0.8):
    """Divisão temporal: reservas mais antigas para treino, mais recentes para teste."""
    df = df.sort_values(["arrival_date", "booking_date"]).reset_index(drop=True)
    corte = int(len(df) * frac_treino)
    treino = df.iloc[:corte].reset_index(drop=True)
    teste = df.iloc[corte:].reset_index(drop=True)
    return treino, teste
