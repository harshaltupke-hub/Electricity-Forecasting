import pandas as pd
import numpy as np


TARGETS = [
    "nat_demand",
    "load_tocumen_mwh",
    "load_santiago_mwh",
    "load_david_mwh"
]


def add_time_features(df):
    df = df.copy()

    # Cyclic hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hourOfDay"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hourOfDay"] / 24)

    # Cyclic day of week
    df["dow_sin"] = np.sin(2 * np.pi * (df["dayOfWeek"] - 1) / 7)
    df["dow_cos"] = np.cos(2 * np.pi * (df["dayOfWeek"] - 1) / 7)

    # Seasonal features
    month = df["datetime"].dt.month
    day_of_year = df["datetime"].dt.dayofyear

    df["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

    df["doy_sin"] = np.sin(2 * np.pi * (day_of_year - 1) / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * (day_of_year - 1) / 365.25)

    # Peak indicators
    df["daytime_peak"] = df["hourOfDay"].between(10, 15).astype(int)
    df["evening_peak"] = df["hourOfDay"].between(18, 20).astype(int)

    # Calendar interactions
    df["hour_weekend"] = df["hourOfDay"] * df["weekend"]
    df["hour_holiday"] = df["hourOfDay"] * df["holiday"]

    return df


def add_demand_features(df):
    df = df.copy()

    nodes = ["national", "tocumen", "santiago", "david"]

    for node in nodes:
        w2 = f"{node}_week_X-2_mwh"
        w3 = f"{node}_week_X-3_mwh"
        w4 = f"{node}_week_X-4_mwh"
        ma = f"{node}_MA_X-4_mwh"

        # Weekly trends
        df[f"{node}_trend_2_3"] = df[w2] - df[w3]
        df[f"{node}_trend_3_4"] = df[w3] - df[w4]
        df[f"{node}_trend_2_4"] = df[w2] - df[w4]

        # Deviation from historical moving average
        df[f"{node}_dev_ma"] = df[w2] - df[ma]

        # Relative deviation
        df[f"{node}_ratio_ma"] = df[w2] / (df[ma] + 1e-6)

    return df


def add_weather_features(df):
    df = df.copy()

    # Temperature × humidity
    df["toc_temp_humidity"] = df["T2M_toc"] * df["QV2M_toc"]
    df["san_temp_humidity"] = df["T2M_san"] * df["QV2M_san"]
    df["dav_temp_humidity"] = df["T2M_dav"] * df["QV2M_dav"]

    # Nonlinear temperature
    df["toc_temp_sq"] = df["T2M_toc"] ** 2
    df["san_temp_sq"] = df["T2M_san"] ** 2
    df["dav_temp_sq"] = df["T2M_dav"] ** 2

    # Cross-zone temperature differences
    df["temp_toc_san_diff"] = df["T2M_toc"] - df["T2M_san"]
    df["temp_toc_dav_diff"] = df["T2M_toc"] - df["T2M_dav"]
    df["temp_san_dav_diff"] = df["T2M_san"] - df["T2M_dav"]

    return df


def add_stability_features(df):
    df = df.copy()

    nodes = ["national", "tocumen", "santiago", "david"]

    for node in nodes:
        w2 = f"{node}_week_X-2_mwh"
        w3 = f"{node}_week_X-3_mwh"
        w4 = f"{node}_week_X-4_mwh"

        weekly = df[[w2, w3, w4]]

        df[f"{node}_weekly_mean"] = weekly.mean(axis=1)
        df[f"{node}_weekly_std"] = weekly.std(axis=1)
        df[f"{node}_weekly_range"] = (
            weekly.max(axis=1) - weekly.min(axis=1)
        )

        df[f"{node}_weekly_cv"] = (
            df[f"{node}_weekly_std"]
            / (df[f"{node}_weekly_mean"] + 1e-6)
        )

        df[f"{node}_trend_consistency"] = (
            np.sign(df[w2] - df[w3])
            + np.sign(df[w3] - df[w4])
        )

    return df


def add_net_load_features(df):
    df = df.copy()

    # Net-load proxies based only on information available as input
    df["net_load_ma_proxy"] = (
        df["national_MA_X-4_mwh"]
        - df["total_vre_gen_mwh"]
    )

    df["net_load_w2_proxy"] = (
        df["national_week_X-2_mwh"]
        - df["total_vre_gen_mwh"]
    )

    df["net_load_w3_proxy"] = (
        df["national_week_X-3_mwh"]
        - df["total_vre_gen_mwh"]
    )

    df["net_load_w4_proxy"] = (
        df["national_week_X-4_mwh"]
        - df["total_vre_gen_mwh"]
    )

    df["vre_penetration_proxy"] = (
        df["total_vre_gen_mwh"]
        / (df["national_MA_X-4_mwh"] + 1e-6)
    )

    df["solar_wind_balance"] = (
        df["total_solar_gen_mwh"]
        - df["total_wind_gen_mwh"]
    )

    return df


def build_features(df):
    """
    Apply the accepted feature-engineering pipeline.
    """
    df = df.copy()

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        format="%d-%m-%Y %H:%M"
    )

    df = add_time_features(df)
    df = add_demand_features(df)
    df = add_weather_features(df)
    df = add_stability_features(df)
    df = add_net_load_features(df)

    return df


def get_model_features(df):
    """
    Return the 110 accepted model features.
    """
    excluded = {
        "datetime",
        "split",
        "window_id",
        "horizon_hour",
        *TARGETS
    }

    return [col for col in df.columns if col not in excluded]