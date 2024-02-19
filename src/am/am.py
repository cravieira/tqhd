from abc import ABC, abstractmethod
import math
import torch
import torchhd
import logging

from .learning import Centroid
from .prediction import Normal

def _copy_torch_dict(a, b):
    """
    Copyt the state dict of model A to B. This function is useful when
    downcasting a pytorch Module.
    """
    # Copy registered buffers and parameters from the AMMap. Use
    # strict=False to copy only the parameters that AMMap declares.
    # Otherwise, pytorch will also require parameters defined by
    # AMThermometer.
    b.load_state_dict(a.state_dict(), strict=False)
    return b

class BaseAM(ABC, torch.nn.Module):
    """
    Implements an Associative Memory for different VSA models and data
    types.
    """
    def __init__(
            self,
            num_classes,
            dtype,
            learning=Centroid(),
            prediction=Normal(),
            **kwargs
            ):
        super().__init__()
        self.num_classes = num_classes
        self.dtype = dtype
        self.learning = learning
        self.prediction = prediction
        self.am = self.register_buffer('am', None)

    @abstractmethod
    def train_am(self):
        pass

    @abstractmethod
    def search(self, query: torch.Tensor) -> torch.Tensor:
        """
        Search the AM for the most similar vector to query.
        """
        pass

    @abstractmethod
    def add(self, input: torch.Tensor, idx: torch.Tensor):
        pass

    @abstractmethod
    def sub(self, input: torch.Tensor, idx: torch.Tensor):
        pass

    def update(self, input: torch.Tensor, idx: torch.Tensor, retrain=False):
        self.learning.update(self, input, idx, retrain)

    def prediction_am(self):
        """
        Returns an AM that can be used in prediction. This function returns an
        AM with different values to the original AM depending on the prediction
        strategy used. For example, if a faulty AM strategy is evaluated, then
        this function returns a faulty AM.
        """
        return self.prediction.apply(self)

class AMMap(BaseAM):
    """
    Associative Memory for MAP VSAs.
    """
    def __init__(
            self,
            dim,
            num_classes,
            dtype=torch.get_default_dtype(),
            device=None,
            **kwargs
            ):
        self.vsa = 'MAP'
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(num_classes, dtype, **kwargs)
        self.weight = torch.zeros((num_classes, dim), **factory_kwargs)
        self.am = self.weight
        self.dim = dim

    def train_am(self):
        """
        Finish AM train and enable it to execute searches.
        """
        self.am = self.weight

    def search(self, query: torch.Tensor):
        """
        Search the AM for the most similar vector to query.
        """
        #query = torchhd.hard_quantize(query)
        logit = torchhd.cosine_similarity(query, self.am)
        #logit = torchhd.hamming_similarity(query, self.am)
        return logit

    def add(self, input: torch.Tensor, idx: torch.Tensor):
        """
        Add the input tensors to the AM class.
        """
        #input = torchhd.hard_quantize(input)
        self.weight[idx] = self.weight[idx] + input

    def sub(self, input: torch.Tensor, idx: torch.Tensor):
        """
        Sub the input tensors from the given AM classes.
        """
        self.weight[idx] = self.weight[idx] - input

class _Accumulator():
    """
    Helper class to implement BSC majority without needing to keep all operand
    vectors.
    """
    def __init__(self, dim, device=None):
        self.dim = dim
        self.elements = 0
        self.acc = torch.zeros((dim), dtype=torch.long, device=device)
        self._need_update = True # Control flag
        self._cache = None # Bundled tensor

    def add(self, input: torch.Tensor):
        """docstring for add"""
        self.acc = self.acc + input
        self.elements += 1
        self._need_update = True

    def sub(self, input: torch.Tensor):
        """docstring for add"""
        self.acc = self.acc - input
        self.elements -= 1
        self._need_update = True

    def _maj(self):
        """docstring for _maj"""
        n = self.elements

        count = self.acc
        generator=None

        # TODO: Solve bug when maj() is called several times in a row. Maybe
        # the problem is in "count = self.acc" passing a reference and not
        # copying the underlying tensor. So when "count += tiebreaker" is
        # executed, the "self.acc" is changed forever.
        # add a tiebreaker when there are an even number of hvs
        #if n % 2 == 0:
        #    tiebreaker = torch.empty_like(count)
        #    tiebreaker.bernoulli_(0.5, generator=generator)
        #    count += tiebreaker
        #    n += 1

        threshold = n // 2
        return torch.greater(count, threshold).to(torch.int8)

    def maj(self):
        """docstring for maj"""
        # Return a copy of the cached tensor instead of a reference to an
        # inner attribute.
        if not self._need_update:
            return self._cache.clone()

        # Recompute cache and return a copy to it.
        self._cache = self._maj()
        return self._cache.clone()

class AMBsc(BaseAM):
    """
    Associative Memory for BSC VSAs.
    """
    def __init__(
            self,
            dim,
            num_classes,
            dtype=torch.get_default_dtype(),
            device=None,
            **kwargs
            ):
        self.vsa = 'BSC'
        super().__init__(num_classes, dtype, **kwargs)
        self.accumulators = [_Accumulator(dim, device=device) for _ in range(num_classes)]
        self.am = None

    def train_am(self):
        """
        Finish AM train and enable it to execute searches.
        """
        tensors = [a.maj() for a in self.accumulators]
        self.am = torch.vstack(tensors)

    def search(self, query: torch.Tensor):
        """
        Search the AM for the most similar vector to query.
        """
        am = self.prediction_am()
        logit = torchhd.hamming_similarity(query, am)
        return logit

    def add(self, input: torch.Tensor, idx: torch.Tensor):
        """
        Add the input tensors to the AM class.
        """
        #for i in range(len(idx)):
        #    self.accumulators[idx[i]].add(input[i])
        # Non-batched implementation
        self.accumulators[idx].add(input)

    def sub(self, input: torch.Tensor, idx: torch.Tensor):
        """
        Sub the input tensors from the given AM classes.
        """
        #for i in range(len(idx)):
        #    self.accumulators[idx[i]].add(input[i])
        # Non-batched implementation
        self.accumulators[idx].add(torch.logical_not(input))

# Quantization AMs #
# These AMs are experimental models that perform quantization. The goal is to
# take a query vector from the MAP model and map it to a binary model.
def normalize(t, dtype=torch.get_default_dtype()):
    # Normalize input
    dot = torch.sum(t*t, dim=-1, dtype=dtype)
    mag = torch.sqrt(dot)
    # Adjust shape
    mag = mag.unsqueeze(1)
    #norm = t/mag
    norm = torch.div(t, mag)

    return norm

def _create_poles(input: torch.Tensor, intervals, quantile=0.25):
    """
    Create quantization poles used by the AM object in its transformation
    based on the input. This function assumes that the input is normalized.
    """
    # Create 1D tensor with quantization poles
    # Start linspace at the furtherst quantile
    qth = quantile
    q = torch.tensor([qth, 1-qth])
    q_vals = input.quantile(q)
    # Pick the value furtherst from 0
    start = torch.max(torch.abs(q_vals))
    poles = torch.linspace(-start.item(), start.item(), steps=intervals)
    return poles

def _find_poles(input: torch.Tensor, poles: torch.Tensor):
    """
    Find the closest quantization pole to the given input. Returns a tensor
    of indices to the closest poles.
    """
    # Make poles a tensor of [Intervals, 1, 1 ..., 1] depending on the
    # shape of the input.
    view = [1] *len(input.shape)
    poles = poles.view(-1, *view)
    # Get absolute difference to each pole
    diffs = torch.abs(input - poles)
    _, inds = torch.min(diffs, dim=0)
    return inds

def _map_to_binary_encoding(input: torch.Tensor, inds, table: torch.Tensor):
    """
    Map the given input to the encoding table according to inds.
    """
    input_dim = input.shape[-1]
    table_entry_dim = table.shape[-1]
    res = table[inds].view((-1, input_dim*table_entry_dim))
    return res

class AMThermometer(AMMap):
    """
    Quantizes vector to Thermometer patterns.
    """
    def __init__(
            self,
            dim,
            num_classes,
            bits,
            intervals,
            dtype=torch.get_default_dtype(),
            device=None,
            **kwargs):
        self.possible_encodings = bits + 1
        if intervals > self.possible_encodings:
            raise RuntimeError(f'Number of intervals ({intervals}) is too big for the number of bits given ({bits}). Only {self.possible_encodings} are possible.')

        super().__init__(dim, num_classes, dtype, device, **kwargs)
        self.bits = bits
        self.intervals = intervals
        self.register_buffer('enc_table', None)
        self.register_buffer('poles', None)
        self.enc_table = self._encode_table(bits, intervals)
        self.poles = None

    def _encode_table(self, bits, entries):
        """
        Create the encode table based on a Thermometer encoding.
        """
        all_combinations = bits+1
        t = torchhd.thermometer(all_combinations, bits, vsa='BSC')

        enc_table = t
        # Remove middle entries if the number of required entries is less than
        # the number of thermometer encodings available.
        if entries < all_combinations:
            rows = entries//2 # Number of top and below rows used
            top_rows = t[..., 0:rows, :]
            bot_rows = t[..., -rows:, :]

            # Create the final table without the rows in the middle
            enc_table = torch.vstack([top_rows, bot_rows])
        # enc_table is a BSC vector and it cannot be used together with MAP,
        # otherwise torchhd throws an exception when indexing. Convert the
        # generated themometer table to MAP.
        enc_table = torchhd.ensure_vsa_tensor(enc_table, vsa='MAP', dtype=torch.int32)
        return enc_table

    @abstractmethod
    def _create_poles(self, input) -> torch.Tensor:
        """docstring for _create_poles"""
        pass

    def _get_poles(self, input: torch.Tensor) -> torch.Tensor:
        """
        Return the poles tensor used by this AM. Creates the quantization poles
        values if they do not exist yet based on the given input. Assumes the
        input is normalized.
        """
        if self.poles is None:
            self.poles = self._create_poles(input)
        return self.poles

        # Alternative approach generating the quantizaiton poles to each input
        #return self._create_poles(input, self.quantile)

    def transform(self, input):
        t = torchhd.functional.ensure_vsa_tensor(input, vsa='MAP', dtype=torch.float)
        dtype = torch.float

        norm = normalize(t, dtype=dtype)

        poles = self._get_poles(norm)
        inds = _find_poles(norm, poles)

        # Create plain binary vectors
        res = _map_to_binary_encoding(norm, inds, self.enc_table)
        return res

    def train_am(self):
        super().train_am()
        self.am = self.transform(self.am)

    def search(self, query):
        vector = self.transform(query)
        am = self.prediction_am()
        logit = torchhd.hamming_similarity(vector, am)
        return logit

class AMThermometerQuantile(AMThermometer):
    """
    Quantizes vector to Thermometer patterns. This class sets the quantization
    poles based on the quantile values of the first transformed vectors.
    """
    def __init__(
            self,
            dim,
            num_classes,
            bits,
            intervals,
            quantile,
            dtype=torch.get_default_dtype(),
            device=None,
            **kwargs):
        super().__init__(dim, num_classes, bits, intervals, dtype=dtype, device=device, **kwargs)
        self.quantile = quantile
        self.poles = None

    def _create_poles(self, input: torch.Tensor):
        """docstring for _create_poles"""
        return _create_poles(input, self.intervals, self.quantile)

class AMThermometerDeviation(AMThermometer):
    """
    Quantizes vector to Thermometer patterns. This class sets the quantization
    poles based on the standard deviation of the input vectors.
    """
    def __init__(
            self,
            dim,
            num_classes,
            bits,
            intervals,
            deviation=1.0,
            dtype=torch.get_default_dtype(),
            device=None,
            **kwargs):
        super().__init__(dim, num_classes, bits, intervals, dtype=dtype, device=device, **kwargs)
        self.deviation = deviation
        self.poles = None

    @classmethod
    def from_AMMap(cls, a: AMMap, bits: int, intervals: int, deviation: float, **kwargs):
        dim = a.dim
        num_classes = a.num_classes
        b_obj = cls(
                dim=dim,
                num_classes=num_classes,
                bits=bits,
                deviation=deviation,
                intervals=intervals,
                **kwargs
                )
        _copy_torch_dict(a, b_obj)
        # Make sure the AMThermometer has an AM to search
        b_obj.train_am()
        return b_obj

    def _create_poles(self, input: torch.Tensor):
        """
        Create quantization poles used by the AM object in its transformation
        based on the input. This function assumes that the input is normalized.
        """
        # Create 1D tensor with quantization poles
        # Start linspace at -std_deviation*multiplier
        intervals = self.intervals
        multiplier = self.deviation
        dev = torch.std(input)
        start = dev*multiplier
        poles = torch.linspace(-start.item(), start.item(), steps=intervals)
        print(poles)
        return poles

class AMSignQuantize(AMMap):
    """
    Implement an AM MAP that predicts using sign quantization. Every
    vector is transformed to binary and the similarity is measured with
    hamming distance.
    """
    def __init__(
            self,
            dim,
            num_classes,
            dtype=torch.get_default_dtype(),
            device=None,
            **kwargs):
        super().__init__(
            dim,
            num_classes,
            dtype=dtype,
            device=device,
            **kwargs)

    @classmethod
    def from_AMMap(cls, a: AMMap, **kwargs):
        dim = a.dim
        num_classes = a.num_classes
        b_obj = cls(
                dim=dim,
                num_classes=num_classes,
                **kwargs
                )
        _copy_torch_dict(a, b_obj)
        b_obj.train_am()
        return b_obj

    def transform(self, input: torch.Tensor) -> torchhd.VSATensor:
        """
        Sign quantize the input vectors.
        """
        sign_quantized = torchhd.hard_quantize(input)
        binary = torch.where(sign_quantized > 0, 1, 0)
        bsc = torchhd.ensure_vsa_tensor(binary, vsa='BSC')
        return bsc

    def train_am(self):
        super().train_am()
        self.am = self.transform(self.am)

    def search(self, query: torch.Tensor):
        """
        Search the AM for the most similar vector for the query.
        """
        vector = self.transform(query)
        logit = torchhd.hamming_similarity(vector, self.am)
        return logit

class PQHDC(AMMap):
    """
    Implement the quantization technique PQ-HDC based on the paper "PQ-HDC:
    Projection-Based Quantization Scheme for Flexible and Efficient
    Hyperdimensional Computing". PQ-HDC trains on bipolar sign quantized MAP
    vectors. At the end of the training phase, it binarizes its trained MAP AM
    to multiple binary AMs. Thus, each MAP vector becomes one or more BSC
    vector. The BSC vectors are generated by projection and receive a weight.
    The similarity measurement is  taken by a Pseudo Hamming Distance (PSD)
    that takes in consideration the weight of each binarized vector.
    """
    def __init__(
            self,
            dim,
            num_classes,
            projections,
            dtype=torch.get_default_dtype(),
            device=None,
            **kwargs):
        super().__init__(dim, num_classes, dtype, device, **kwargs)
        self.projections = projections
        self.register_buffer('weights', None)
        self.weights = None

    @classmethod
    def from_AMMap(cls, a: AMMap, projections: int, **kwargs):
        logging.warning(
            'Transforming a MAP AM to PQHDC can lead to different results than '
            'the original paper since the paper suggests the sign quantization '
            'of query vectors also in training (Fig. 2), whereas it is not '
            'obligatory in AMMap training.')
        dim = a.dim
        num_classes = a.num_classes
        b_obj = cls(
                dim=dim,
                num_classes=num_classes,
                projections=projections,
                **kwargs
                )
        _copy_torch_dict(a, b_obj)
        # Make sure the AMThermometer has an AM to search
        b_obj.train_am()
        return b_obj

    def bipolarize(self, input):
        """
        Returns a sign quantized bipolar tensor. The values in input are mapped
        to {-1, 1}.
        """
        return torchhd.hard_quantize(input)

    def binarize(self, input):
        """
        Returns a sign quantized bnary tensor. The values in input are mapped
        to {0, 1}.
        """
        sign_quantized = torchhd.hard_quantize(input)
        binary = torch.where(sign_quantized > 0, 1, 0)
        bsc = torchhd.ensure_vsa_tensor(binary, vsa='BSC')
        return bsc

    def orthogonal_projection(self, a, b):
        """
        Returns a vector corresponding the orthogonal projection of a onto b.
        """
        b_length = torch.linalg.norm(b, ord=2)
        dot = torch.dot(a, b)
        return b*dot/(b_length**2)

    def _linear_regression(self, X, Y):
        '''
        Returns the w_class used in the Algorithm 1.
        '''
        X_t = X.t()
        inv = None
        squared = X @ X_t
        try:
            inv = torch.inverse(squared)
            P = X_t @ inv
            w_class = Y@P
        except:
            logging.warn("Failed to invert matrix, torch.linalg.lstsq")
            w_class = torch.linalg.lstsq(X_t, Y).solution
        return w_class

    def transform(self, am: torch.Tensor, projections):
        """
        Transform a MAP AM into a projection-based bipolarized AM. This
        function implements the Algorithm 1 of the paper.
        """
        classes = am.shape[0]
        D = am.shape[1] # Number of dimensions in the AM
        AMbip = [None] * classes # Final bipolarized (or binarized) AM
        weights = [None] * classes

        # Use "L <number>" to reference the Algorithm presented in the paper.
        # L 1
        for c in range(classes):
            bips = []
            # L 3
            hv_loss = am[c]

            # For each projection
            # L 4
            for i in range(projections):
                # L 5
                bip = self.bipolarize(hv_loss)
                # L 6
                ortho_proj = self.orthogonal_projection(hv_loss, bip)
                # L 7
                hv_loss = hv_loss - ortho_proj
                # L 8
                bips.append(bip)

            # Extra aliases and boilerplate code
            bips = torch.vstack(bips)
            AMbip[c] = bips
            X = bips

            # L 9
            Y = am[c]
            # L 10
            # L 11
            w_class = self._linear_regression(X, Y)
            w_class = w_class.view(1,-1).t()
            # L 12
            length = torch.linalg.norm(w_class*bips)
            w_class = math.sqrt(D) * w_class / length

            # Extra command! The original algorithm does not pow the w_class
            # vector even though it is necessary since Algorithm 2 relies that
            # the sum of elements in w_class integrate to 1, so that the PHD
            # returns a number between 0 and D. By squaring w_class, we get the
            # linear contribution of each projection to the total D.
            w_class = w_class.pow(2)
            # w_class is in the shape [projections, 1], make it a flat tensor.
            weights[c] = w_class.flatten()

        AMbip = torch.stack(AMbip)
        # Return a binarized AM with values in {0, 1}, and not a bipolar {-1, 1}
        AMbin = self.binarize(AMbip)
        weights = torch.stack(weights)

        return AMbin, weights

    def train_am(self):
        super().train_am()
        self.am, self.weights = self.transform(self.am, self.projections)

    def phs(self, query, am, weights):
        """
        Implements Pseudo Hamming Similarity (PHS). This is analogous to the
        Pseudo Hamming Distance (PHD) used in the paper.
        """
        # Implement PHS because torchhd only has hamming similarity available
        # out of the box, and both metrics are analogous since HS = dim - HD
        classes = am.shape[0]
        projections = weights.shape[-1]

        # Originally, Algorithm 2, L 6 adds extra constant terms like -D/2 and
        # D/2. This code does not use them since they are irrelevant for final
        # accuracy because they contribute equally to each PHD class.
        hamm_sim = torchhd.hamming_similarity(query, am)
        # hamm_sim shape is [classes, 1, projections]. Eliminate the midle axis.
        hamm_sim = hamm_sim.view((classes, projections))
        weighted_sim = weights * hamm_sim
        sims = torch.sum(weighted_sim, axis=1)

        # Make sims shape [1, classes] since this is the uniform shape used in
        # the test routine.
        sims.unsqueeze_(dim=0)
        return sims

    def search(self, query):
        vector = self.binarize(query)
        am = self.prediction_am()
        logit = self.phs(vector, am, self.weights)
        return logit

    # Intercept calls to add and sub when training the model to train using
    # hard quantized vectors.
    def add(self, input: torch.Tensor, idx: torch.Tensor):
        v = self.bipolarize(input)
        super().add(v, idx)

    def sub(self, input: torch.Tensor, idx: torch.Tensor):
        v = self.bipolarize(input)
        super().sub(v, idx)
