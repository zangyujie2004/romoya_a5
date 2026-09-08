from typing import Dict, Callable, Union, List
import zarr
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

def dict_apply(
        x: Dict[str, torch.Tensor], 
        func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result

def dict_apply_reduce(
        x: List[Dict[str, torch.Tensor]],
        reduce_func: Callable[[List[torch.Tensor]], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key in x[0].keys():
        result[key] = reduce_func([x_[key] for x_ in x])
    return result

def _normalize(x, params, forward=True):
    assert "scale" in params
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    scale = params["scale"]
    offset = params["offset"]
    x = x.to(device=scale.device, dtype=scale.dtype)
    src_shape = x.shape
    x = x.reshape(-1, scale.shape[0])
    if forward:
        x = x * scale + offset
    else:
        x = (x - offset) / scale
    x = x.reshape(src_shape)
    return x

def _fit(
    data: Union[torch.Tensor, np.ndarray, zarr.Array],
    last_n_dims=1,
    dtype=torch.float32,
    mode="limits",
    output_max=1.0,
    output_min=-1.0,
    range_eps=1e-4,
    fit_offset=True,
):
    assert mode in ["limits", "gaussian"]
    assert last_n_dims >= 0
    assert output_max > output_min

    # convert data to torch and type
    if isinstance(data, zarr.Array):
        data = data[:]
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    if dtype is not None:
        data = data.type(dtype)

    # convert shape
    dim = 1
    if last_n_dims > 0:
        dim = np.prod(data.shape[-last_n_dims:])
    data = data.reshape(-1, dim)

    # compute input stats min max mean std
    input_min, _ = data.min(axis=0)
    input_max, _ = data.max(axis=0)
    input_mean = data.mean(axis=0)
    input_std = data.std(axis=0)

    # compute scale and offset
    if mode == "limits":
        if fit_offset:
            # unit scale
            input_range = input_max - input_min
            ignore_dim = input_range < range_eps
            input_range[ignore_dim] = output_max - output_min
            scale = (output_max - output_min) / input_range
            offset = output_min - scale * input_min
            offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
            # ignore dims scaled to mean of output max and min
        else:
            # use this when data is pre-zero-centered.
            assert output_max > 0
            assert output_min < 0
            # unit abs
            output_abs = min(abs(output_min), abs(output_max))
            input_abs = torch.maximum(torch.abs(input_min), torch.abs(input_max))
            ignore_dim = input_abs < range_eps
            input_abs[ignore_dim] = output_abs
            # don't scale constant channels
            scale = output_abs / input_abs
            offset = torch.zeros_like(input_mean)
    elif mode == "gaussian":
        ignore_dim = input_std < range_eps
        scale = input_std.clone()
        scale[ignore_dim] = 1
        scale = 1 / scale

        if fit_offset:
            offset = -input_mean * scale
        else:
            offset = torch.zeros_like(input_mean)

    # save
    this_params = nn.ParameterDict(
        {
            "scale": scale,
            "offset": offset,
            "input_stats": nn.ParameterDict(
                {
                    "min": input_min,
                    "max": input_max,
                    "mean": input_mean,
                    "std": input_std,
                }
            ),
        }
    )
    for p in this_params.parameters():
        p.requires_grad_(False)
    return this_params

class DictOfTensorMixin(nn.Module):
    def __init__(self, params_dict=None):
        super().__init__()
        if params_dict is None:
            params_dict = nn.ParameterDict()
        self.params_dict = params_dict

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        def dfs_add(dest, keys, value: torch.Tensor):
            if len(keys) == 1:
                dest[keys[0]] = value
                return

            if keys[0] not in dest:
                dest[keys[0]] = nn.ParameterDict()
            dfs_add(dest[keys[0]], keys[1:], value)

        def load_dict(state_dict, prefix):
            out_dict = nn.ParameterDict()
            for key, value in state_dict.items():
                value: torch.Tensor
                if key.startswith(prefix):
                    param_keys = key[len(prefix):].split('.')[1:]
                    # if len(param_keys) == 0:
                    #     import pdb; pdb.set_trace()
                    dfs_add(out_dict, param_keys, value.clone())
            return out_dict

        self.params_dict = load_dict(state_dict, prefix + 'params_dict')
        self.params_dict.requires_grad_(False)
        return 

class LinearNormalizer(DictOfTensorMixin):
    avaliable_modes = ["limits", "gaussian"]

    @torch.no_grad()
    def fit(
        self,
        data: Union[Dict, torch.Tensor, np.ndarray, zarr.Array],
        last_n_dims=1,
        dtype=torch.float32,
        mode="limits",
        output_max=1.0,
        output_min=-1.0,
        range_eps=1e-4,
        fit_offset=True,
    ):
        if isinstance(data, dict):
            for key, value in data.items():
                self.params_dict[key] = _fit(
                    value,
                    last_n_dims=last_n_dims,
                    dtype=dtype,
                    mode=mode,
                    output_max=output_max,
                    output_min=output_min,
                    range_eps=range_eps,
                    fit_offset=fit_offset,
                )
        else:
            self.params_dict["_default"] = _fit(
                data,
                last_n_dims=last_n_dims,
                dtype=dtype,
                mode=mode,
                output_max=output_max,
                output_min=output_min,
                range_eps=range_eps,
                fit_offset=fit_offset,
            )

    def __call__(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self.normalize(x)

    def __getitem__(self, key: str):
        return SingleFieldLinearNormalizer(self.params_dict[key])

    def __setitem__(self, key: str, value: "SingleFieldLinearNormalizer"):
        self.params_dict[key] = value.params_dict

    def _normalize_impl(self, x, forward=True):
        if isinstance(x, dict):
            result = dict()
            for key, value in x.items():
                params = self.params_dict[key]
                result[key] = _normalize(value, params, forward=forward)
            return result
        else:
            if "_default" not in self.params_dict:
                raise RuntimeError("Not initialized")
            params = self.params_dict["_default"]
            return _normalize(x, params, forward=forward)

    def normalize(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self._normalize_impl(x, forward=True)

    def unnormalize(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self._normalize_impl(x, forward=False)

    def get_input_stats(self) -> Dict:
        if len(self.params_dict) == 0:
            raise RuntimeError("Not initialized")
        if len(self.params_dict) == 1 and "_default" in self.params_dict:
            return self.params_dict["_default"]["input_stats"]

        result = dict()
        for key, value in self.params_dict.items():
            if key != "_default":
                result[key] = value["input_stats"]
        return result

    def get_output_stats(self, key="_default"):
        input_stats = self.get_input_stats()
        if "min" in input_stats:
            # no dict
            return dict_apply(input_stats, self.normalize)

        result = dict()
        for key, group in input_stats.items():
            this_dict = dict()
            for name, value in group.items():
                this_dict[name] = self.normalize({key: value})[key]
            result[key] = this_dict
        return result

class SingleFieldLinearNormalizer(DictOfTensorMixin):
    avaliable_modes = ["limits", "gaussian"]

    @torch.no_grad()
    def fit(
        self,
        data: Union[torch.Tensor, np.ndarray, zarr.Array],
        last_n_dims=1,
        dtype=torch.float32,
        mode="limits",
        output_max=1.0,
        output_min=-1.0,
        range_eps=1e-4,
        fit_offset=True,
    ):
        self.params_dict = _fit(
            data,
            last_n_dims=last_n_dims,
            dtype=dtype,
            mode=mode,
            output_max=output_max,
            output_min=output_min,
            range_eps=range_eps,
            fit_offset=fit_offset,
        )

    @classmethod
    def create_fit(cls, data: Union[torch.Tensor, np.ndarray, zarr.Array], **kwargs):
        obj = cls()
        obj.fit(data, **kwargs)
        return obj

    @classmethod
    def create_manual(
        cls,
        scale: Union[torch.Tensor, np.ndarray],
        offset: Union[torch.Tensor, np.ndarray],
        input_stats_dict: Dict[str, Union[torch.Tensor, np.ndarray]],
    ):
        def to_tensor(x):
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x)
            x = x.flatten()
            return x

        # check
        for x in [offset] + list(input_stats_dict.values()):
            assert x.shape == scale.shape
            assert x.dtype == scale.dtype

        params_dict = nn.ParameterDict(
            {
                "scale": to_tensor(scale),
                "offset": to_tensor(offset),
                "input_stats": nn.ParameterDict(
                    dict_apply(input_stats_dict, to_tensor)
                ),
            }
        )
        return cls(params_dict)

    @classmethod
    def create_identity(cls, dtype=torch.float32):
        scale = torch.tensor([1], dtype=dtype)
        offset = torch.tensor([0], dtype=dtype)
        input_stats_dict = {
            "min": torch.tensor([-1], dtype=dtype),
            "max": torch.tensor([1], dtype=dtype),
            "mean": torch.tensor([0], dtype=dtype),
            "std": torch.tensor([1], dtype=dtype),
        }
        return cls.create_manual(scale, offset, input_stats_dict)

    def normalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=True)

    def unnormalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=False)

    def get_input_stats(self):
        return self.params_dict["input_stats"]

    def get_output_stats(self):
        return dict_apply(self.params_dict["input_stats"], self.normalize)

    def __call__(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self.normalize(x)

def get_range_normalizer_from_stat(stat, output_max=1, output_min=-1, range_eps=1e-7):
    # -1, 1 normalization
    input_max = stat['max']
    input_min = stat['min']
    input_range = input_max - input_min
    ignore_dim = input_range < range_eps
    input_range[ignore_dim] = output_max - output_min
    scale = (output_max - output_min) / input_range
    offset = output_min - scale * input_min
    offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]

    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )

def get_identity_normalizer_from_stat(stat):
    scale = np.ones_like(stat['min'])
    offset = np.zeros_like(stat['min'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )

def array_to_stats(arr: np.ndarray):
    stat = {
        'min': np.min(arr, axis=0),
        'max': np.max(arr, axis=0),
        'mean': np.mean(arr, axis=0),
        'std': np.std(arr, axis=0)
    }
    return stat

def concatenate_normalizer(normalizers: list):
    scale = torch.concatenate([normalizer.params_dict['scale'] for normalizer in normalizers], axis=-1)
    offset = torch.concatenate([normalizer.params_dict['offset'] for normalizer in normalizers], axis=-1)
    input_stats_dict = dict_apply_reduce(
        [normalizer.params_dict['input_stats'] for normalizer in normalizers],
        lambda x: torch.concatenate(x,axis=-1))
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=input_stats_dict
    )

# Merged robot vector (T, 62): joint | eef_L | eef_R | effort | vel
# eef per arm: xyz(3) + rot6d(6) + gripper(1) = 10
ROBOT_MERGED_62_SEGMENTS: tuple[tuple[slice, str], ...] = (
    (slice(0, 14), "limits"),    # joint L+R
    (slice(14, 17), "limits"),   # eef L xyz
    (slice(17, 23), "identity"), # eef L rot6d
    (slice(23, 24), "limits"),   # eef L gripper
    (slice(24, 27), "limits"),   # eef R xyz
    (slice(27, 33), "identity"), # eef R rot6d
    (slice(33, 34), "limits"),   # eef R gripper
    (slice(34, 48), "limits"),   # effort L+R
    (slice(48, 62), "limits"),   # vel L+R
)

ROBOT_STORAGE_KEYS = ("state_30hz", "action_30hz", "state_120hz", "action_120hz")
TACTILE_STORAGE_KEY = "tactile"


def get_robot_merged_normalizer(
    values: np.ndarray,
    *,
    segments: tuple[tuple[slice, str], ...] = ROBOT_MERGED_62_SEGMENTS,
    output_max: float = 1.0,
    output_min: float = -1.0,
    range_eps: float = 1e-7,
) -> SingleFieldLinearNormalizer:
    """Fit limits/identity normalizer for merged 62-d robot state or action."""
    arr = np.asarray(values, dtype=np.float32).reshape(-1, values.shape[-1])
    expected_dim = max(sl.stop for sl, _ in segments)
    if arr.shape[-1] != expected_dim:
        raise ValueError(f"expected last dim {expected_dim}, got {arr.shape[-1]}")

    normalizers = []
    for sl, mode in segments:
        stat = array_to_stats(arr[:, sl])
        if mode == "limits":
            normalizers.append(
                get_range_normalizer_from_stat(stat, output_max=output_max, output_min=output_min, range_eps=range_eps)
            )
        elif mode == "identity":
            normalizers.append(get_identity_normalizer_from_stat(stat))
        else:
            raise ValueError(f"unknown segment mode: {mode}")
    return concatenate_normalizer(normalizers)


def compute_tactile_channel_stats(
    data: Union[torch.Tensor, np.ndarray, zarr.Array],
    *,
    channels_per_point: int = 6,
    chunk_size: int = 512,
) -> dict[str, np.ndarray]:
    """Stream min/max/mean/std over tactile pointcloud, pooled per 6-channel semantics."""
    if isinstance(data, zarr.Array):
        total = int(data.shape[0])
    else:
        data = np.asarray(data, dtype=np.float32)
        total = int(data.shape[0])

    n_total = 0
    sum_x = None
    sum_x2 = None
    min_x = None
    max_x = None

    for start in range(0, total, chunk_size):
        end = min(total, start + chunk_size)
        if isinstance(data, zarr.Array):
            chunk = np.asarray(data[start:end], dtype=np.float32)
        else:
            chunk = data[start:end]

        x = chunk.reshape(-1, channels_per_point)
        if x.shape[0] == 0:
            continue

        chunk_sum = x.sum(axis=0)
        chunk_sum2 = (x * x).sum(axis=0)
        chunk_min = x.min(axis=0)
        chunk_max = x.max(axis=0)

        if sum_x is None:
            sum_x = chunk_sum
            sum_x2 = chunk_sum2
            min_x = chunk_min
            max_x = chunk_max
        else:
            sum_x += chunk_sum
            sum_x2 += chunk_sum2
            min_x = np.minimum(min_x, chunk_min)
            max_x = np.maximum(max_x, chunk_max)
        n_total += x.shape[0]

    if n_total == 0:
        raise ValueError("no tactile samples found while computing normalizer stats")

    mean_x = sum_x / float(n_total)
    var_x = np.maximum(sum_x2 / float(n_total) - mean_x * mean_x, 0.0)
    return {
        "min": min_x,
        "max": max_x,
        "mean": mean_x,
        "std": np.sqrt(var_x),
    }


def tile_normalizer_channels(normalizer: SingleFieldLinearNormalizer, repeat: int) -> SingleFieldLinearNormalizer:
    """Repeat per-point channel stats across merged tactile bundles (e.g. 6 -> 24)."""
    if repeat < 1:
        raise ValueError(f"repeat must be >= 1, got {repeat}")
    scale = normalizer.params_dict["scale"]
    offset = normalizer.params_dict["offset"]
    stats = normalizer.params_dict["input_stats"]
    return SingleFieldLinearNormalizer.create_manual(
        scale=torch.tile(scale, (repeat,)),
        offset=torch.tile(offset, (repeat,)),
        input_stats_dict={name: torch.tile(value, (repeat,)) for name, value in stats.items()},
    )


def get_tactile_merged_normalizer(
    data: Union[torch.Tensor, np.ndarray, zarr.Array],
    *,
    num_bundles: int = 4,
    channels_per_point: int = 6,
    chunk_size: int = 512,
    output_max: float = 1.0,
    output_min: float = -1.0,
    range_eps: float = 1e-7,
) -> SingleFieldLinearNormalizer:
    """Min-max normalize tactile pointcloud channels, tiled for merged storage last dim."""
    stat = compute_tactile_channel_stats(
        data,
        channels_per_point=channels_per_point,
        chunk_size=chunk_size,
    )
    base = get_range_normalizer_from_stat(
        stat,
        output_max=output_max,
        output_min=output_min,
        range_eps=range_eps,
    )
    return tile_normalizer_channels(base, repeat=num_bundles)


def get_action_normalizer(actions: np.ndarray, temporally_independent_normalization=False):
    assert not temporally_independent_normalization, "Not use temporally independent normalization now"
    assert len(actions.shape) == 2 or len(actions.shape) == 3
    if not temporally_independent_normalization:
        actions = actions.reshape(-1, actions.shape[-1])

    D = actions.shape[-1]
    normalizers = []
    normalizers.append(get_range_normalizer_from_stat(array_to_stats(actions[...,:3])))
    # don't normalize rotation
    normalizers.append(get_identity_normalizer_from_stat(array_to_stats(actions[...,3:9])))
    if D == 10:
        normalizers.append(get_range_normalizer_from_stat(array_to_stats(actions[...,9:])))

    normalizer = concatenate_normalizer(normalizers)
    return normalizer

def get_force_normalizer(values: np.ndarray):
    assert len(values.shape) in [2, 3]
    return SingleFieldLinearNormalizer.create_fit(values)

def get_normalizer(data_list):
    action_normalize_list = [
        'abs_action', 'relative_action', 'state',
        'abs_action_4x', 'relative_action_4x', 'state_4x'
    ]
    force_normalize_list = [
        'force', 'force_4x',
        'tac_force', 'tac_force_4x'
    ]
    tactile_normalize_list = [
        'tactile', 'tactile_4x'
    ]
    normalize_keys = set(
        action_normalize_list + force_normalize_list + tactile_normalize_list
    )

    all_data_dict = defaultdict(list)
    for d in data_list:
        for k in normalize_keys:
            if k in d:
                all_data_dict[k].append(d[k])

    normalizer = LinearNormalizer()
    for key, value_list in all_data_dict.items():
        arr = np.stack(value_list, axis=0)
        if key in action_normalize_list:
            normalizer[key] = get_action_normalizer(arr)
        elif key in force_normalize_list:
            normalizer[key] = get_force_normalizer(arr)
        elif key in tactile_normalize_list:
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(arr)

    return normalizer

def replace_normalizer(normalizer1, normalizer2, normalizer1_path):
    src = normalizer2.params_dict["tactile"]
    dst = normalizer1.params_dict["tactile"]

    with torch.no_grad():
        dst["input_stats"]["min"].copy_(src["input_stats"]["min"])
        dst["input_stats"]["max"].copy_(src["input_stats"]["max"])
        dst["input_stats"]["mean"].copy_(src["input_stats"]["mean"])
        dst["input_stats"]["std"].copy_(src["input_stats"]["std"])

        dst["offset"].copy_(src["offset"])
        dst["scale"].copy_(src["scale"])
    
    torch.save(normalizer1, normalizer1_path)

if __name__ == "__main__":
    normalizer1_path = '/data/datasets/tro_tac/data_samples_demo/adj_demo/tacwm_samples_32_downsample_1/normalizer.pth'
    normalizer1 = torch.load('/data/datasets/tro_tac/data_samples_demo/adj_demo/tacwm_samples_32_downsample_1/normalizer.pth')
    normalizer2 = torch.load('/data/datasets/tro_tac/data_samples/tactile_all/tacwm_samples_32/normalizer_copy.pth')
    replace_normalizer(normalizer1, normalizer2, normalizer1_path)
    print('end')