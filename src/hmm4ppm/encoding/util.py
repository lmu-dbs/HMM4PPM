from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer
import numpy as np
import pandas as pd

from ..util.logging import init_logging

logger = init_logging(__name__, 'encoding.log')

class StrictlyNonNegativeOrdinalEncoder(OrdinalEncoder):
    
    UNKNOWN = -2
    MISSING = -1
    SHIFT = 2
    
    def __init__(self):
        super().__init__(handle_unknown='use_encoded_value', 
                         unknown_value=self.UNKNOWN, 
                         encoded_missing_value=self.MISSING)
        
    def fit(self, X, y = None):
        super().fit(X)
        self.n_categories_ = [len(c) for c in self.categories_]
        return self
    
    def transform(self, X):
        raw = super().transform(X)
        shifted = raw + self.SHIFT
        return shifted
    
    def fit_transform(self, X, y = None):
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X):
        X = np.asarray(X)
        raw = X - self.SHIFT
        return super().inverse_transform(raw)
    
    def num_categories_per_col(self):
        return [n + self.SHIFT for n in self.n_categories_]
    
class GammaScaler():
    """Scale Gamma distributed features by scaling them with their mean
    """

    def __init__(self, clamping_val: float = 1e-12):
        
        self.clamping_val = clamping_val
    
        self.means = None
        self.n_features_in = None
        
    def _reset(self):
        self.means = None
        self.n_features_in = None

    def _validate_data(self, X):
        """Checks if the data provided fits Gamma assumptions (support of [0,+inf] with density 0 at value 0)
        """
        if isinstance(X, pd.DataFrame):
            validated_data = X.to_numpy().astype(float)
        elif isinstance(X, pd.Series):
            validated_data = X.to_numpy().reshape(-1, 1).astype(float)
        elif isinstance(X, np.ndarray):
            validated_data = X.copy().astype(float)
        else:
            raise ValueError("X needs to be of type pd.DataFrame, pd.Series or np.array")
        
        offending_values = validated_data <= 0
        
        if offending_values.sum() > 0:
            offending_features = offending_values.sum(axis=0) > 0
            logger.warning(f"Feature values in features {[idx for idx, _ in enumerate(offending_features) if _]} provided do not fulfill Gamma assumptions (support of [0,+inf] with density 0 at value 0) - clamping to {self.clamping_val} before applying scaling")
            
            validated_data[offending_values] = self.clamping_val
            return validated_data
        else:
            return X

    def fit(self, X, y=None, validate: int = True):
        """Compute the mean used for later scaling.
        """
        self._reset()
        if validate:
            validated_data = self._validate_data(X)
        else:
            validated_data = X
        
        n_features = validated_data.shape[1]
        means = validated_data.mean(axis=0)
        
        self.n_features_in = n_features
        self.means = means
    
    def transform(self, X):
        
        validated_data = self._validate_data(X)
        scaled_data = validated_data / self.means
        
        return scaled_data
    
    def fit_transform(self, X):
        self._reset()
        validated_data = self._validate_data(X)
        self.fit(validated_data, validate = False)
        scaled_data = self.transform(validated_data)
        return scaled_data
    
    def inverse_transform(self, X):
        
        retransformed_data = X * self.means
        
        return retransformed_data
    
class ZeroInflatedGammaScaler():
    """Scale ZeroInflated/Gamma distributed features by scaling them with their mean
    """

    def __init__(self):
        
        self.clamping_val = 0
        
        self.means = None
        self.n_features_in = None
        
        self.mins = None
        self.maxs = None
        
    def _reset(self):
        self.means = None
        self.n_features_in = None

    def _validate_data(self, X):
        """Checks if the data provided fits ZeroInflated/Gamma assumptions (support of [0,+inf])
        """
        if isinstance(X, pd.DataFrame):
            validated_data = X.to_numpy().astype(float)
        elif isinstance(X, pd.Series):
            validated_data = X.to_numpy().reshape(-1, 1).astype(float)
        elif isinstance(X, np.ndarray):
            validated_data = X.copy().astype(float)
        else:
            raise ValueError("X needs to be of type pd.DataFrame, pd.Series or np.array")
        
        offending_values = validated_data < 0
        
        if offending_values.sum() > 0:
            offending_features = offending_values.sum(axis=0) > 0
            logger.warning(f"Feature values in features {[idx for idx, _ in enumerate(offending_features) if _]} provided do not fulfill Gamma assumptions (support of [0,+inf]) - clamping to {self.clamping_val} before applying scaling")
            
            validated_data[offending_values] = self.clamping_val
            return validated_data
        else:
            return X

    def fit(self, X, y=None, validate: int = True):
        """Compute the mean used for later scaling.
        """
        self._reset()
        if validate:
            validated_data = self._validate_data(X)
        else:
            validated_data = X
        
        n_features = validated_data.shape[1]
        
        mins = validated_data.min(axis=0)
        maxs = validated_data.max(axis=0)
        
        # means_with_zeros = validated_data.mean(axis=0)
        means_without_zeros = validated_data[validated_data > 0].mean(axis=0)
        
        self.n_features_in = n_features
        self.means = means_without_zeros
        self.mins = mins
        self.maxs = maxs
    
    def transform(self, X):
        
        validated_data = self._validate_data(X)
        scaled_data = validated_data / self.means
        
        return scaled_data
    
    def fit_transform(self, X):
        self._reset()
        validated_data = self._validate_data(X)
        self.fit(validated_data, validate = False)
        scaled_data = self.transform(validated_data)
        return scaled_data
    
    def inverse_transform(self, X):
        
        retransformed_data = X * self.means
        
        return retransformed_data
    
class EncodingFactory:
    _encoding = {
        "OneHotEncoder": OneHotEncoder,
        "OrdinalEncoder": OrdinalEncoder,
        "StrictlyNonNegativeOrdinalEncoder": StrictlyNonNegativeOrdinalEncoder,
        "StandardScaler": StandardScaler,
        "GammaScaler": GammaScaler,
        "ZeroInflatedGammaScaler": ZeroInflatedGammaScaler,
        "RobustScaler": RobustScaler,
        "MinMaxScaler": MinMaxScaler,
    }

    @classmethod
    def create(cls, encoding, **encoding_params):
        if encoding in cls._encoding:
            encoding_class = cls._encoding[encoding]
            return encoding_class(**encoding_params)
        else:
            raise ValueError(f"Unknown encoding: {encoding}")

class TransformFactory:
    _transform = {
        "PowerTransformer": PowerTransformer
    }

    @classmethod
    def create(cls, transform, **transform_params):
        if transform in cls._transform:
            transform_class = cls._transform[transform]
            return transform_class(**transform_params)
        else:
            raise ValueError(f"Unknown transform: {transform}")

class Decoding:

    def __init__(self, encoders: dict):
        self.encoders = encoders

    def decode_sample(self, encoded_sample) -> dict:

        cols_for_encoder = dict()
        for encoder, cols in self.encoders.values():
            if len(cols) > 0:
                if isinstance(encoder, OneHotEncoder):
                    ncols_cols_tpl = (sum([len(enc_cats) for enc_cats in encoder.categories_]), cols)
                elif isinstance(encoder, OrdinalEncoder) or isinstance(encoder, StandardScaler) or isinstance(encoder, RobustScaler) or isinstance(encoder, MinMaxScaler):
                    ncols_cols_tpl = (len(cols), cols)
                else:
                    raise NotImplementedError(f"encoder type {type(encoder)} not implemented for decoding")
                cols_for_encoder[encoder] = ncols_cols_tpl
            
        decoded_sample_dict = dict()

        start_index = 0
        for encoder, (n_decode_cols, decode_cols) in cols_for_encoder.items():
            if n_decode_cols > 0:
                sample_to_decode = encoded_sample[:, start_index:(start_index+n_decode_cols)]
                
                decoded_sample = encoder.inverse_transform(sample_to_decode)

                for dict_col, decoded_col in zip(decode_cols, decoded_sample[0,:]):
                    decoded_sample_dict[dict_col] = decoded_col

                start_index += n_decode_cols

        return decoded_sample_dict
    
    def decode_samples(self, encoded_samples) -> dict:

        decoded_samples = [self.decode_sample(sample) for sample in encoded_samples]

        # transform list of single value dicts to dict of lists per encoded attribute
        decoded_samples_list_dict = dict()

        # pull attribute keys from first decoded sample and restructure object
        for attribute in decoded_samples[0].keys():
            decoded_samples_list_dict[attribute] = [sample[attribute] for sample in decoded_samples]

        return decoded_samples_list_dict
    
    def decode_sample_sequences(self, encoded_sample_sequences) -> dict:

        decoded_sample_sequences = [self.decode_samples(sample_seq) for sample_seq in encoded_sample_sequences]

        # transform list of single value dicts to dict of lists per encoded attribute
        decoded_sample_sequences_list_dict = dict()

        # pull attribute keys from first decoded sample and restructure object
        for attribute in decoded_sample_sequences[0].keys():
            decoded_sample_sequences_list_dict[attribute] = [sequence[attribute] for sequence in decoded_sample_sequences]

        return decoded_sample_sequences_list_dict


class Retransformation:

    def __init__(self, transformers: dict):
        self.transformers = transformers

    def retransform_sample(self, transformed_sample) -> dict:

        cols_for_transformer = dict()
        for transformer, cols in self.transformers.values():
            if len(cols) > 0:
                if isinstance(transformer, PowerTransformer):
                    ncols_cols_tpl = (len(cols), cols)
                else:
                    raise NotImplementedError(f"transformer type {type(transformer)} not implemented for decoding")
                cols_for_transformer[transformer] = ncols_cols_tpl
            
        retransformed_sample_dict = dict()

        start_index = 0
        for transformer, (n_retransform_cols, retransform_cols) in cols_for_transformer.items():
            if n_retransform_cols > 0:
                sample_to_retransform = transformed_sample[:, start_index:(start_index+n_retransform_cols)]
                
                decoded_sample = transformer.inverse_transform(pd.DataFrame(sample_to_retransform, columns=retransform_cols))

                for dict_col, decoded_col in zip(retransform_cols, decoded_sample[0,:]):
                    retransformed_sample_dict[dict_col] = decoded_col

                start_index += n_retransform_cols

        return retransformed_sample_dict
    
    def retransform_samples(self, transformed_samples) -> dict:

        if isinstance(transformed_samples, dict):
            rearranged_transformed_samples = self.rearrange_samples(transformed_samples)
        else:
            rearranged_transformed_samples = transformed_samples.copy()

        retransformed_samples = [self.retransform_sample(sample) for sample in rearranged_transformed_samples]

        # transform list of single value dicts to dict of lists per encoded attribute
        retransformed_samples_list_dict = dict()

        # pull attribute keys from first decoded sample and restructure object
        for attribute in retransformed_samples[0].keys():
            retransformed_samples_list_dict[attribute] = [sample[attribute] for sample in retransformed_samples]

        if isinstance(transformed_samples, dict):
            transformed_samples.update(retransformed_samples_list_dict)
            retransformed_samples_list_dict = transformed_samples

        return retransformed_samples_list_dict
    
    def rearrange_samples(self, transformed_samples) -> list:
        all_transform_cols = list()
        for (_, cols) in self.transformers.values():
            all_transform_cols.extend(cols)
        transformed_sample_cols = list()
        for col in all_transform_cols:
            transformed_sample_cols.append(np.array(transformed_samples[col]))
        rearranged_transformed_samples = np.stack(transformed_sample_cols).transpose()
        rearranged_transformed_samples = [np.array([sample]) for sample in rearranged_transformed_samples]

        return rearranged_transformed_samples

    def rearrange_sample_sequences(self, transformed_sample_sequences) -> list:
        all_transform_cols = list()
        for (_, cols) in self.transformers.values():
            all_transform_cols.extend(cols)
        rearranged_transformed_sample_sequences = list()
        for sequence_idx in range(len(transformed_sample_sequences[all_transform_cols[0]])):
            rearranged_sample_sequence = list()
            for col in all_transform_cols:
                rearranged_sample_sequence.append(np.array(transformed_sample_sequences[col][sequence_idx]))
            rearranged_sample_sequence = np.stack(rearranged_sample_sequence).transpose()
            rearranged_transformed_sample_sequences.append([np.array([sample]) for sample in rearranged_sample_sequence])
        
        return rearranged_transformed_sample_sequences

    def retransform_sample_sequences(self, transformed_sample_sequences) -> dict:

        rearranged_transformed_sample_sequences = self.rearrange_sample_sequences(transformed_sample_sequences)
        
        retransformed_sample_sequences = [self.retransform_samples(sample_seq) for sample_seq in rearranged_transformed_sample_sequences]

        # transform list of single value dicts to dict of lists per encoded attribute
        retransformed_sample_sequences_list_dict = dict()

        # pull attribute keys from first decoded sample and restructure object
        for attribute in retransformed_sample_sequences[0].keys():
            retransformed_sample_sequences_list_dict[attribute] = [sequence[attribute] for sequence in retransformed_sample_sequences]

        transformed_sample_sequences.update(retransformed_sample_sequences_list_dict)
        retransformed_sample_sequences_list_dict = transformed_sample_sequences

        return retransformed_sample_sequences_list_dict


if __name__=='__main__':
    
    gamma_scaler = GammaScaler()
    
    data = np.array([[-1,1,1,3,3,3,2,3,1,10], [34,24,23,23,2,4,44,23,11,13], [33,36,45,33,13,2,2,3,1,25]]).T
    
    gamma_scaler.fit(data)
    
    scaled_data = gamma_scaler.transform(data)
    
    scaled_data_ft = gamma_scaler.fit_transform(data)
    
    retransformed_data = gamma_scaler.inverse_transform(scaled_data)
    
    scaled_data == scaled_data_ft
    
    retransformed_data == data