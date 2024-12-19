from typing import Union, Iterable, Optional

import numpy as np
import pandas as pd

from mlfinlab.data_structures.base_bars import BaseBars


class StandardBars(BaseBars):
    """
    Creates standard bars (tick, volume, or dollar), triggered when a specified cumulative metric threshold is reached.
    """

    def __init__(self, metric: str, threshold: int = 50000, batch_size: int = 20000000, additional_features=None):
        """
        Constructor

        :param metric: (str) Type of bar to create. Options: 'cum_ticks', 'cum_volume', 'cum_dollar_value'.
        :param threshold: (int) Threshold at which a new bar is formed.
        :param batch_size: (int) Number of rows to read in from the csv, per batch.
        :param additional_features: (list) Additional feature computation objects.
        """
        super().__init__(metric=metric, batch_size=batch_size, additional_features=additional_features)
        self.threshold = threshold

    def _reset_cache(self):
        """
        Reset the cache when a new bar is formed.
        """
        self.open_price = None
        self.high_price, self.low_price = -np.inf, np.inf
        self.cum_statistics = {'cum_ticks': 0, 'cum_dollar_value': 0, 'cum_volume': 0, 'cum_buy_volume': 0}

    def _extract_bars(self, data: Union[list, tuple, np.ndarray]) -> list:
        """
        Iterate over ticks and form bars once the cumulative metric reaches the threshold.

        :param data: (array-like) of shape (n, 3): [date_time, price, volume]
        :return: (list) Extracted bars.
        """
        list_bars = []

        for row in data:
            # Update ticks for additional features
            self._update_ticks_in_bar(row)

            date_time = row[0]
            self.tick_num += 1
            price = float(row[1])
            volume = row[2]
            dollar_value = price * volume
            signed_tick = self._apply_tick_rule(price)

            if self.open_price is None:
                self.open_price = price

            # Update high/low
            self.high_price, self.low_price = self._update_high_low(price)

            # Update cumulative statistics
            self.cum_statistics['cum_ticks'] += 1
            self.cum_statistics['cum_dollar_value'] += dollar_value
            self.cum_statistics['cum_volume'] += volume
            if signed_tick == 1:
                self.cum_statistics['cum_buy_volume'] += volume

            # Check if threshold met
            if self.cum_statistics[self.metric] >= self.threshold:
                # Compute additional features before creating the bar
                self._compute_additional_features()

                # Create bar
                self._create_bars(date_time, price, self.high_price, self.low_price, list_bars)

                # Reset feature and tick caches
                self._reset_ticks_in_bar()
                self._reset_computed_additional_features()

                # Reset cache for next bar
                self._reset_cache()

        return list_bars


def get_dollar_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], threshold: float = 70000000,
                    batch_size: int = 20000000,
                    verbose: bool = True, to_csv: bool = False, output_path: Optional[str] = None,
                    additional_features=None):
    """
    Creates dollar bars with optional additional features.
    """
    bars = StandardBars(metric='cum_dollar_value', threshold=threshold, batch_size=batch_size,
                        additional_features=additional_features)
    dollar_bars = bars.batch_run(file_path_or_df=file_path_or_df, verbose=verbose, to_csv=to_csv, output_path=output_path)
    return dollar_bars


def get_volume_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], threshold: float = 70000000,
                    batch_size: int = 20000000,
                    verbose: bool = True, to_csv: bool = False, output_path: Optional[str] = None,
                    additional_features=None):
    """
    Creates volume bars with optional additional features.
    """
    bars = StandardBars(metric='cum_volume', threshold=threshold, batch_size=batch_size,
                        additional_features=additional_features)
    volume_bars = bars.batch_run(file_path_or_df=file_path_or_df, verbose=verbose, to_csv=to_csv, output_path=output_path)
    return volume_bars


def get_tick_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], threshold: float = 70000000,
                  batch_size: int = 20000000,
                  verbose: bool = True, to_csv: bool = False, output_path: Optional[str] = None,
                  additional_features=None):
    """
    Creates tick bars with optional additional features.
    """
    bars = StandardBars(metric='cum_ticks', threshold=threshold, batch_size=batch_size,
                        additional_features=additional_features)
    tick_bars = bars.batch_run(file_path_or_df=file_path_or_df, verbose=verbose, to_csv=to_csv, output_path=output_path)
    return tick_bars


class RelativePriceBars(BaseBars):
    """
    Creates bars whenever price moves a certain percentage (threshold) away from the open_price of the current bar.
    """

    def __init__(self, threshold: float, batch_size: int = 20000000, additional_features=None):
        """
        Constructor

        :param threshold: (float) Relative threshold, e.g., 0.01 for 1% move.
        :param batch_size: (int) Rows read per batch
        :param additional_features: (list) Feature computation objects
        """
        super().__init__(metric='rel_price', batch_size=batch_size, additional_features=additional_features)
        self.threshold = threshold
        self._reset_cache()

    def _reset_cache(self):
        """
        Reset cache for new bar.
        """
        self.open_price = None
        self.high_price, self.low_price = -np.inf, np.inf
        self.cum_statistics = {
            'cum_ticks': 0,
            'cum_dollar_value': 0,
            'cum_volume': 0,
            'cum_buy_volume': 0
        }

    def _extract_bars(self, data: Union[list, tuple, np.ndarray]) -> list:
        """
        Iterate over rows and create bars when price moves beyond the relative threshold from the open_price.
        """
        list_bars = []

        for row in data:
            # Update ticks for additional features
            self._update_ticks_in_bar(row)

            date_time = row[0]
            self.tick_num += 1
            price = float(row[1])
            volume = row[2]
            dollar_value = price * volume
            signed_tick = self._apply_tick_rule(price)

            # Set open price if not set
            if self.open_price is None:
                self.open_price = price

            # Update high/low
            self.high_price, self.low_price = self._update_high_low(price)

            # Update cumulative statistics
            self.cum_statistics['cum_ticks'] += 1
            self.cum_statistics['cum_dollar_value'] += dollar_value
            self.cum_statistics['cum_volume'] += volume
            if signed_tick == 1:
                self.cum_statistics['cum_buy_volume'] += volume

            # Compute threshold bounds
            upper_bound = self.open_price * (1 + self.threshold)
            lower_bound = self.open_price * (1 - self.threshold)

            # Check if price moved beyond threshold
            if price >= upper_bound or price <= lower_bound:
                # Compute additional features before creating the bar
                self._compute_additional_features()

                # Create bar
                self._create_bars(date_time, price, self.high_price, self.low_price, list_bars)

                # Reset feature and tick caches
                self._reset_ticks_in_bar()
                self._reset_computed_additional_features()

                # Reset cache for next bar
                self._reset_cache()

        return list_bars


def get_relative_price_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame],
                            threshold: float,
                            batch_size: int = 20000000,
                            verbose: bool = True,
                            to_csv: bool = False,
                            output_path: Optional[str] = None,
                            additional_features=None) -> pd.DataFrame:
    """
    Creates relative price bars with optional additional features.
    """
    bars = RelativePriceBars(threshold=threshold, batch_size=batch_size, additional_features=additional_features)
    relative_price_bars = bars.batch_run(file_path_or_df=file_path_or_df, verbose=verbose, to_csv=to_csv, output_path=output_path)
    return relative_price_bars
