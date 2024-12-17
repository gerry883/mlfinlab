"""
Inter-bar feature generator which uses trades data and bars index to calculate inter-bar features.

This module defines a class to generate microstructural features after bars have been formed.

Available Built-in Features for Selection (pass these strings in `selected_features`):
-------------------------------------------------------------------------------------
- "avg_tick_size"    : Average trade size within the bar.
- "tick_rule_sum"    : Sum of signed ticks based on price movements.
- "vwap"             : Volume Weighted Average Price.
- "kyle_lambda"      : Kyle's Lambda - measures price impact of trades.
- "amihud_lambda"    : Amihud's Lambda - illiquidity measure using log returns and volume.
- "hasbrouck_lambda" : Hasbrouck's Lambda - liquidity measure using signed ticks and log returns.

Entropy features can be selected individually. If not specified (and selected_features=None),
all of them are computed by default:

Tick-rule Entropies:
- "tick_rule_shannon_entropy"
- "tick_rule_plug_in_entropy"
- "tick_rule_lempel_ziv_entropy"
- "tick_rule_konto_entropy"

Volume Entropies (if volume_encoding is provided):
- "volume_shannon_entropy"
- "volume_plug_in_entropy"
- "volume_lempel_ziv_entropy"
- "volume_konto_entropy"

Pct Entropies (if pct_encoding is provided):
- "pct_shannon_entropy"
- "pct_plug_in_entropy"
- "pct_lempel_ziv_entropy"
- "pct_konto_entropy"

Custom (Additional) Features:
-----------------------------
Users can provide a list of custom feature objects to `additional_features`, each having a `compute()` method
that takes a DataFrame of the current bar's ticks and returns a computed value.
"""

import pandas as pd
import numpy as np
from mlfinlab.microstructural_features.entropy import get_shannon_entropy, get_plug_in_entropy, get_lempel_ziv_entropy, \
    get_konto_entropy
from mlfinlab.microstructural_features.encoding import encode_array
from mlfinlab.microstructural_features.second_generation import get_trades_based_kyle_lambda, \
    get_trades_based_amihud_lambda, get_trades_based_hasbrouck_lambda
from mlfinlab.microstructural_features.misc import get_avg_tick_size, vwap
from mlfinlab.microstructural_features.encoding import encode_tick_rule_array
from mlfinlab.util.misc import crop_data_frame_in_batches


class MicrostructuralFeaturesGeneratorEnhanced:
    """
    Class used to generate inter-bar features after bars are formed. Supports selecting built-in features and
    custom user-defined features, as well as separate selection of each entropy type.

    :param trades_input: (str or pd.DataFrame) Path to the csv file or DataFrame containing [date_time, price, volume].
    :param tick_num_series: (pd.Series) Series of tick numbers where bars were formed.
    :param batch_size: (int) Number of rows to read in from the csv, per batch.
    :param volume_encoding: (dict) Encoding scheme for trade sizes (for volume-based entropy).
    :param pct_encoding: (dict) Encoding scheme for log returns (for pct-based entropy).
    :param selected_features: (list) List of built-in features to compute. If None, all are computed.
                              For entropies, specify individually:
                              - tick_rule_shannon_entropy, tick_rule_plug_in_entropy,
                                tick_rule_lempel_ziv_entropy, tick_rule_konto_entropy
                              - volume_shannon_entropy, volume_plug_in_entropy,
                                volume_lempel_ziv_entropy, volume_konto_entropy (if volume_encoding is given)
                              - pct_shannon_entropy, pct_plug_in_entropy,
                                pct_lempel_ziv_entropy, pct_konto_entropy (if pct_encoding is given)
    :param additional_features: (list) List of custom feature objects with a `compute()` method.
    """

    def __init__(self, trades_input: (str, pd.DataFrame), tick_num_series: pd.Series, batch_size: int = 2e7,
                 volume_encoding: dict = None, pct_encoding: dict = None,
                 selected_features: list = None, additional_features: list = None):

        if isinstance(trades_input, str):
            self.generator_object = pd.read_csv(trades_input, chunksize=batch_size, parse_dates=[0])
            # Read in the first row & assert format
            first_row = pd.read_csv(trades_input, nrows=1)
            self._assert_csv(first_row)
        elif isinstance(trades_input, pd.DataFrame):
            self.generator_object = crop_data_frame_in_batches(trades_input, batch_size)
        else:
            raise ValueError('trades_input is neither string(path to a csv file) nor pd.DataFrame')

        # Base properties
        self.tick_num_generator = iter(tick_num_series)
        self.current_bar_tick_num = self.tick_num_generator.__next__()

        # Cache properties
        self.price_diff = []
        self.trade_size = []
        self.tick_rule = []
        self.dollar_size = []
        self.log_ret = []

        # Encodings
        self.volume_encoding = volume_encoding
        self.pct_encoding = pct_encoding

        # Batch_run properties
        self.prev_price = None
        self.prev_tick_rule = 0
        self.tick_num = 0

        # Selected features and additional features
        self.selected_features = selected_features
        self.additional_features = additional_features
        self.computed_additional_features = []

    def get_features(self, verbose=True, to_csv=False, output_path=None):
        """
        Reads a CSV file or DataFrame in batches and constructs corresponding microstructural features.
        
        If selected_features is None, all features (including all entropy measures) are computed.

        :param verbose: (bool) Whether to print progress during batch processing.
        :param to_csv: (bool) If True, write results to CSV; if False, return a DataFrame.
        :param output_path: (str) Path to the output CSV file if to_csv=True.
        :return: (pd.DataFrame or None) DataFrame of computed features or None if written to CSV.
        """

        if to_csv is True:
            header = True
            open(output_path, 'w').close()  # Clean output file

        # Determine columns dynamically
        cols = ['date_time']
        # Basic features
        if self._is_selected('avg_tick_size'):
            cols.append('avg_tick_size')
        if self._is_selected('tick_rule_sum'):
            cols.append('tick_rule_sum')
        if self._is_selected('vwap'):
            cols.append('vwap')
        if self._is_selected('kyle_lambda'):
            cols += ['kyle_lambda', 'kyle_lambda_t_value']
        if self._is_selected('amihud_lambda'):
            cols += ['amihud_lambda', 'amihud_lambda_t_value']
        if self._is_selected('hasbrouck_lambda'):
            cols += ['hasbrouck_lambda', 'hasbrouck_lambda_t_value']

        # Entropy columns (tick_rule)
        if self._entropy_selected('tick_rule_shannon_entropy'):
            cols.append('tick_rule_shannon_entropy')
        if self._entropy_selected('tick_rule_plug_in_entropy'):
            cols.append('tick_rule_plug_in_entropy')
        if self._entropy_selected('tick_rule_lempel_ziv_entropy'):
            cols.append('tick_rule_lempel_ziv_entropy')
        if self._entropy_selected('tick_rule_konto_entropy'):
            cols.append('tick_rule_konto_entropy')

        # Volume entropy columns
        if self.volume_encoding is not None:
            if self._entropy_selected('volume_shannon_entropy'):
                cols.append('volume_shannon_entropy')
            if self._entropy_selected('volume_plug_in_entropy'):
                cols.append('volume_plug_in_entropy')
            if self._entropy_selected('volume_lempel_ziv_entropy'):
                cols.append('volume_lempel_ziv_entropy')
            if self._entropy_selected('volume_konto_entropy'):
                cols.append('volume_konto_entropy')

        # Pct entropy columns
        if self.pct_encoding is not None:
            if self._entropy_selected('pct_shannon_entropy'):
                cols.append('pct_shannon_entropy')
            if self._entropy_selected('pct_plug_in_entropy'):
                cols.append('pct_plug_in_entropy')
            if self._entropy_selected('pct_lempel_ziv_entropy'):
                cols.append('pct_lempel_ziv_entropy')
            if self._entropy_selected('pct_konto_entropy'):
                cols.append('pct_konto_entropy')

        # Additional features
        if self.additional_features:
            for i, _ in enumerate(self.additional_features, start=1):
                cols.append(f'additional_feature_{i}')

        final_bars = []
        count = 0

        # Process in batches
        for batch in self.generator_object:
            if verbose:
                print('Batch number:', count)

            list_bars, stop_flag = self._extract_bars(data=batch)

            if to_csv:
                pd.DataFrame(list_bars, columns=cols).to_csv(output_path, header=header, index=False, mode='a')
                header = False
            else:
                final_bars += list_bars

            count += 1
            if stop_flag:
                break

        if final_bars:
            return pd.DataFrame(final_bars, columns=cols)
        return None

    def _is_selected(self, feature_name):
        # If selected_features is None, all features are selected
        return self.selected_features is None or feature_name in self.selected_features

    def _entropy_selected(self, feature_name):
        # If selected_features is None, we consider all entropies selected
        if self.selected_features is None:
            # Means all features including entropies are selected by default
            return True
        return feature_name in self.selected_features

    def _reset_cache(self):
        self.price_diff = []
        self.trade_size = []
        self.tick_rule = []
        self.dollar_size = []
        self.log_ret = []
        self._reset_computed_additional_features()

    def _extract_bars(self, data):
        list_bars = []
        for row in data.values:
            date_time = row[0]
            price = float(row[1])
            volume = row[2]
            dollar_value = price * volume
            signed_tick = self._apply_tick_rule(price)

            self.tick_num += 1

            # Derivative variables
            price_diff = self._get_price_diff(price)
            log_ret = self._get_log_ret(price)

            self.price_diff.append(price_diff)
            self.trade_size.append(volume)
            self.tick_rule.append(signed_tick)
            self.dollar_size.append(dollar_value)
            self.log_ret.append(log_ret)

            self.prev_price = price

            if self.tick_num >= self.current_bar_tick_num:
                self._get_bar_features(date_time, list_bars)
                try:
                    self.current_bar_tick_num = self.tick_num_generator.__next__()
                except StopIteration:
                    return list_bars, True
                self._reset_cache()
        return list_bars, False

    def _get_bar_features(self, date_time, list_bars):
        features = [date_time]

        # Basic Features
        if self._is_selected('avg_tick_size'):
            features.append(get_avg_tick_size(self.trade_size))
        if self._is_selected('tick_rule_sum'):
            features.append(sum(self.tick_rule))
        if self._is_selected('vwap'):
            features.append(vwap(self.dollar_size, self.trade_size))

        if self._is_selected('kyle_lambda'):
            features.extend(get_trades_based_kyle_lambda(self.price_diff, self.trade_size, self.tick_rule))
        if self._is_selected('amihud_lambda'):
            features.extend(get_trades_based_amihud_lambda(self.log_ret, self.dollar_size))
        if self._is_selected('hasbrouck_lambda'):
            features.extend(get_trades_based_hasbrouck_lambda(self.log_ret, self.dollar_size, self.tick_rule))

        # Entropy Features
        encoded_tick_rule_message = encode_tick_rule_array(self.tick_rule)

        # Tick rule entropies
        if self._entropy_selected('tick_rule_shannon_entropy'):
            features.append(get_shannon_entropy(encoded_tick_rule_message))
        if self._entropy_selected('tick_rule_plug_in_entropy'):
            features.append(get_plug_in_entropy(encoded_tick_rule_message))
        if self._entropy_selected('tick_rule_lempel_ziv_entropy'):
            features.append(get_lempel_ziv_entropy(encoded_tick_rule_message))
        if self._entropy_selected('tick_rule_konto_entropy'):
            features.append(get_konto_entropy(encoded_tick_rule_message))

        # Volume entropies
        if self.volume_encoding is not None:
            volume_message = encode_array(self.trade_size, self.volume_encoding)
            if self._entropy_selected('volume_shannon_entropy'):
                features.append(get_shannon_entropy(volume_message))
            if self._entropy_selected('volume_plug_in_entropy'):
                features.append(get_plug_in_entropy(volume_message))
            if self._entropy_selected('volume_lempel_ziv_entropy'):
                features.append(get_lempel_ziv_entropy(volume_message))
            if self._entropy_selected('volume_konto_entropy'):
                features.append(get_konto_entropy(volume_message))

        # Pct entropies
        if self.pct_encoding is not None:
            pct_message = encode_array(self.log_ret, self.pct_encoding)
            if self._entropy_selected('pct_shannon_entropy'):
                features.append(get_shannon_entropy(pct_message))
            if self._entropy_selected('pct_plug_in_entropy'):
                features.append(get_plug_in_entropy(pct_message))
            if self._entropy_selected('pct_lempel_ziv_entropy'):
                features.append(get_lempel_ziv_entropy(pct_message))
            if self._entropy_selected('pct_konto_entropy'):
                features.append(get_konto_entropy(pct_message))

        # Compute additional custom features
        custom_features = self._compute_additional_features()
        features.extend(custom_features)

        list_bars.append(features)

    def _compute_additional_features(self):
        computed_features = []
        if self.additional_features:
            tick_df = pd.DataFrame({
                'price_diff': self.price_diff,
                'trade_size': self.trade_size,
                'tick_rule': self.tick_rule,
                'log_ret': self.log_ret,
                'dollar_size': self.dollar_size,
            })
            for feature in self.additional_features:
                computed_features.append(feature.compute(tick_df))
        return computed_features

    def _reset_computed_additional_features(self):
        self.computed_additional_features = []

    def _apply_tick_rule(self, price):
        if self.prev_price is not None:
            tick_diff = price - self.prev_price
        else:
            tick_diff = 0

        if tick_diff != 0:
            signed_tick = np.sign(tick_diff)
            self.prev_tick_rule = signed_tick
        else:
            signed_tick = self.prev_tick_rule

        return signed_tick

    def _get_price_diff(self, price):
        return price - self.prev_price if self.prev_price is not None else 0

    def _get_log_ret(self, price):
        return np.log(price / self.prev_price) if self.prev_price is not None else 0

    @staticmethod
    def _assert_csv(test_batch):
        assert test_batch.shape[1] == 3, 'Must have only 3 columns in csv: date_time, price, & volume.'
        assert isinstance(test_batch.iloc[0, 1], float), 'price column in csv not float.'
        assert not isinstance(test_batch.iloc[0, 2], str), 'volume column in csv not int or float.'

        try:
            pd.to_datetime(test_batch.iloc[0, 0])
        except ValueError:
            print('csv file, column 0, not a date time format:',
                  test_batch.iloc[0, 0])
