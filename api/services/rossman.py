from pathlib import Path
from pickle import load

import numpy as np
import pandas as pd


class Rossman:
    def __init__(self):
        # Define o caminho base de forma dinâmica
        self.home_path = Path(__file__).parent.parent.parent if "__file__" in globals() else Path()
        param_path = self.home_path / "parameters"

        # Carrega os scalers garantindo o fechamento dos arquivos
        with Path.open(param_path / "competition_distance_scaler.pkl", "rb") as f:
            self.rs_competition_distance = load(f)

        with Path.open(param_path / "year_scaler.pkl", "rb") as f:  # Ajustado o nome do arquivo
            self.mms_year = load(f)

        with Path.open(param_path / "competition_time_month_scaler.pkl", "rb") as f:
            self.rs_competition_time_month = load(f)

        with Path.open(param_path / "promo_time_week_scaler.pkl", "rb") as f:
            self.mms_promo_time_week = load(f)

        with Path.open(param_path / "store_type_scaler.pkl", "rb") as f:
            self.le = load(f)

    def limpando_dados(self, df1):
        # 1.1 Renomeando As Colunas
        cols_map = {
            "Store": "store",
            "DayOfWeek": "day_of_week",
            "Date": "date",
            "Sales": "sales",
            "Customers": "customers",
            "Open": "open",
            "Promo": "promo",
            "StateHoliday": "state_holiday",
            "SchoolHoliday": "school_holiday",
            "StoreType": "store_type",
            "Assortment": "assortment",
            "CompetitionDistance": "competition_distance",
            "CompetitionOpenSinceMonth": "competition_open_since_month",
            "CompetitionOpenSinceYear": "competition_open_since_year",
            "Promo2": "promo2",
            "Promo2SinceWeek": "promo2_since_week",
            "Promo2SinceYear": "promo2_since_year",
            "PromoInterval": "promo_interval",
        }
        df1 = df1.rename(columns={k: v for k, v in cols_map.items() if k in df1.columns})

        # 1.3 Tipos De Dados
        df1["date"] = pd.to_datetime(df1["date"])

        # 1.5 Filtrando os NA
        month_map = {
            1: "Jan",
            2: "Feb",
            3: "Mar",
            4: "Apr",
            5: "May",
            6: "Jun",
            7: "Jul",
            8: "Aug",
            9: "Sep",
            10: "Oct",
            11: "Nov",
            12: "Dec",
        }

        df1["competition_distance"] = df1["competition_distance"].fillna(200000)

        df1["competition_open_since_month"] = df1["competition_open_since_month"].fillna(
            df1["date"].dt.month
        )
        df1["competition_open_since_year"] = df1["competition_open_since_year"].fillna(
            df1["date"].dt.year
        )

        df1["promo2_since_week"] = df1["promo2_since_week"].fillna(
            df1["date"].dt.isocalendar().week
        )
        df1["promo2_since_year"] = df1["promo2_since_year"].fillna(df1["date"].dt.year)

        df1["promo_interval"] = df1["promo_interval"].fillna(0)
        df1["month_map"] = df1["date"].dt.month.map(month_map)

        df1["is_promo"] = df1[["promo_interval", "month_map"]].apply(
            lambda x: (
                0
                if x["promo_interval"] == 0
                else (1 if x["month_map"] in str(x["promo_interval"]).split(",") else 0)
            ),
            axis=1,
        )

        # 1.6 Mudando O Tipo Dos Dados
        df1["competition_open_since_month"] = df1["competition_open_since_month"].astype(int)
        df1["competition_open_since_year"] = df1["competition_open_since_year"].astype(int)
        df1["promo2_since_week"] = df1["promo2_since_week"].astype(int)
        df1["promo2_since_year"] = df1["promo2_since_year"].astype(int)

        return df1

    def engenharia_recursos(self, df2):
        # 2.3 Engenharia De Recursos
        # Colunas de tempo
        df2["year"] = df2["date"].dt.year
        df2["month"] = df2["date"].dt.month
        df2["day"] = df2["date"].dt.day
        df2["week_of_year"] = df2["date"].dt.isocalendar().week

        df2["year_week"] = df2["date"].dt.strftime("%Y-%W")

        # Colunas de Competição por tempo
        year = df2["competition_open_since_year"].astype(int).astype(str)
        month = df2["competition_open_since_month"].astype(int).astype(str)

        df2["competition_since"] = pd.to_datetime(year + "-" + month + "-01", errors="coerce")
        df2["competition_time_month"] = (
            (df2["date"] - df2["competition_since"]).dt.days / 30
        ).astype(int)

        # Colunas de Promoção
        df2["date"] = df2["date"].dt.tz_localize(None)
        promo_date_str = (
            df2["promo2_since_year"].astype(str) + "-" + df2["promo2_since_week"].astype(str) + "-1"
        )

        df2["promo2_since"] = pd.to_datetime(promo_date_str, format="%Y-%W-%w") - pd.Timedelta(
            days=7
        )
        df2["promo_time_week"] = ((df2["date"] - df2["promo2_since"]).dt.days / 7).astype(int)

        # Sortimento
        df2["assortment"] = df2["assortment"].apply(
            lambda x: "basic" if x == "a" else "extra" if x == "b" else "extended"
        )

        # Feriados nacionais
        # Mapeamento dos feriados
        HOLIDAY_MAP = {"a": "public_holiday", "b": "easter_holiday", "c": "christmas"}
        df2["state_holiday"] = df2["state_holiday"].map(HOLIDAY_MAP).fillna("regular_day")

        return df2

    def filtrando_variaveis(self, df3):
        # 3.1 Filtragem Das Linhas
        df3 = df3[(df3["open"] != 0) & (df3["sales"] > 0)]
        cols_drop = ["customers", "open", "promo_interval", "month_map"]
        df3 = df3.drop(cols_drop, axis=1)

        return df3

    def preparando_dados(self, df5):
        # 5.2 Redimensionamento
        df5["competition_distance"] = self.rs_competition_distance.transform(
            df5[["competition_distance"]].values
        )
        df5["year"] = self.mms_year.transform(df5[["year"]].values)
        df5["competition_time_month"] = self.rs_competition_time_month.transform(
            df5[["competition_time_month"]].values
        )
        df5["promo_time_week"] = self.mms_promo_time_week.transform(df5[["promo_time_week"]].values)

        # 5.3 Transformação
        # 5.3.1 Codificação
        df5 = pd.get_dummies(df5, prefix=["state_holiday"], columns=["state_holiday"])

        df5["store_type"] = self.le.transform(df5["store_type"])

        assortment_dict = {"basic": 1, "extra": 2, "extended": 3}
        df5["assortment"] = df5["assortment"].map(assortment_dict)

        # 5.3.2 Transformação Da Variável Resposta
        if "sales" in df5.columns:
            df5["sales"] = np.log1p(df5["sales"])

        df5["day_of_week_sin"] = df5["day_of_week"].apply(lambda x: np.sin(x * (2.0 * np.pi / 7)))
        df5["day_of_week_cos"] = df5["day_of_week"].apply(lambda x: np.cos(x * (2.0 * np.pi / 7)))

        df5["month_sin"] = df5["month"].apply(lambda x: np.sin(x * (2.0 * np.pi / 12)))
        df5["month_cos"] = df5["month"].apply(lambda x: np.cos(x * (2.0 * np.pi / 12)))

        df5["day_sin"] = df5["day"].apply(lambda x: np.sin(x * (2.0 * np.pi / 30)))
        df5["day_cos"] = df5["day"].apply(lambda x: np.cos(x * (2.0 * np.pi / 30)))

        df5["week_of_year_sin"] = df5["week_of_year"].apply(
            lambda x: np.sin(x * (2.0 * np.pi / 52))
        )
        df5["week_of_year_cos"] = df5["week_of_year"].apply(
            lambda x: np.cos(x * (2.0 * np.pi / 52))
        )

        return df5

    def selecao_variaveis(self, df6):
        # 6.1 Dividindo O DF Em Treino E Teste
        cols_drop = [
            "week_of_year",
            "day",
            "month",
            "day_of_week",
            "competition_since",
            "year_week",
        ]

        # 6.3 Seleção Manual De Variaveis
        cols_selected_boruta = [
            "store",
            "promo",
            "store_type",
            "assortment",
            "competition_distance",
            "competition_open_since_month",
            "competition_open_since_year",
            "promo2",
            "promo2_since_week",
            "promo2_since_year",
            "competition_time_month",
            "promo_time_week",
            "day_of_week_sin",
            "day_of_week_cos",
            "month_sin",
            "month_cos",
            "day_sin",
            "day_cos",
            "week_of_year_sin",
            "week_of_year_cos",
        ]

        # `date` não entra nas features do modelo (evita divergência treino/serve).
        # `sales` ainda está aqui temporariamente, até corrigirmos o problema 1 (leakage).
        feat_to_add = ["sales"]

        # resultado final
        cols_selected_boruta.extend(feat_to_add)

        return df6[cols_selected_boruta]

    def formatando_dados(self, df_brutos):
        df_formatado = self.limpando_dados(df_brutos)
        df_formatado = self.engenharia_recursos(df_formatado)
        df_formatado = self.preparando_dados(df_formatado)

        return df_formatado

    @staticmethod
    def _model_feature_names(model):
        """Retorna as features esperadas pelo modelo salvo."""
        if hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)

        booster = getattr(model, "get_booster", lambda: None)()
        if booster is not None and getattr(booster, "feature_names", None):
            return list(booster.feature_names)

        return None

    def get_prediction(self, model, original_data, test_data):
        cols_expected = self._model_feature_names(model)
        if cols_expected is None:
            raise ValueError("Não foi possível identificar as features esperadas pelo modelo.")

        if "date" in cols_expected:
            raise ValueError(
                "O modelo carregado ainda foi treinado com a feature 'date'. "
                "Execute o main.ipynb novamente (Passo 10) para gerar um modelo sem 'date'."
            )

        for col in cols_expected:
            if col not in test_data.columns:
                test_data[col] = 0

        pred = model.predict(test_data[cols_expected])

        original_data["prediction"] = np.expm1(pred)

        return original_data.to_json(orient="records", date_format="iso")
