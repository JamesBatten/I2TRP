import torch
import numpy as np
import numpy.typing as npt




def to_tensor(x: npt.NDArray | torch.Tensor | list) -> torch.Tensor:
    if type(x) == np.ndarray or type(x) == list:
        x = torch.tensor(x)
    elif type(x) in [int, float, np.float64, np.float32, np.int64, np.int32]:
        x = torch.tensor([x])
    elif type(x) == torch.Tensor:
        pass
    else:
        raise Exception("unhandled type " + str(type(x)))
    return x


def to_array(x: npt.NDArray | torch.Tensor | list | int | float) -> npt.NDArray:
    if torch.is_tensor(x) or type(x) == list:
        x = x.detach().cpu().numpy()
    elif type(x) in [int, float]:
        x = np.array([x])
    return x


def liftsin(x, oct):
    if type(x) == np.ndarray:
        return np.sin(2 * np.pi * oct * x).astype(np.float32)
    elif type(x) == torch.Tensor:
        return torch.sin(2 * np.pi * oct * x).float()
    raise Exception("unhandled type " + str(type(x)))


def liftcos(x, oct):
    if type(x) == np.ndarray:
        return np.cos(2 * np.pi * oct * x).astype(np.float32)
    elif type(x) == torch.Tensor:
        return torch.cos(2 * np.pi * oct * x).float()
    raise Exception("unhandled type " + str(type(x)))


def lift(x, oct, lifter):
    if lifter == "sin":
        return liftsin(x, oct)
    elif lifter == "cos":
        return liftcos(x, oct)
    raise Exception("unhandled lifter " + str(lifter))


def take_along_axis(x, index, axis):
    if type(x) == np.ndarray:
        return np.take(x, index, axis=axis)
    elif type(x) == torch.Tensor:
        if type(index) == int:
            index = torch.tensor([index]).to(x.device)
        return torch.index_select(
            x, dim=axis, index=index
        ).squeeze(axis)
    raise Exception("unhandled type " + str(type(x)))


def add_octaves(x: torch.Tensor | npt.NDArray[np.float32],
                octaves: list[int],
                dim: int,
                channels: list[int],
                ret_type: str | None = None,
                lifters: list[str] = ['sin', 'cos'],
                mask: torch.Tensor | npt.NDArray[np.float32] | None = None
                ) -> npt.NDArray[np.float32] | torch.Tensor:
    '''
        Arguments

            x               :   float tensor or array
            octaves         :   list of k integers
            dim             :   int
            channels        :   list of ints
            ret_type        :   string
            lifters         :   list of strings
            mask            :   float tensor or array
                                must be same shape as x
                                minus the `dim` dimension

        Returns

            x               :   float tensor or array

        Notes

            Example:

                Input: x tensor of shape (b, 3, h, w) (GXY),
                octaves=[1, 2, 4], dim=1, channels=[1, 2],
                lifters=['sin', 'cos']

                Output: x tensor of shape (b, 15, h, w)
                
                Explanation: X and Y will each be lifted into
                3 sin channels and 3 cos channels respectively.

                3 + 6 + 6 = 15

                `dim` specifies the *single* dimension over which to
                compute the octaves, and `channels` specifies
                the indices of the channels to lift.
    '''
    if mask is not None:
        assert len(mask.shape) == len(x.shape) - 1
        for i in range(len(mask.shape)):
            if i != dim:
                assert mask.shape[i] == x.shape[i]

    new_ch = []
    for c_idx in channels:
        if c_idx >= x.shape[dim]:
            raise Exception(
                "channel index " + str(c_idx) + " out of range"
            )
        channel = take_along_axis(x, c_idx, dim)
        for oct in octaves:
            for lft in lifters:
                new_ch.append(lift(channel, oct, lft))
    if type(x) == np.ndarray:
        new_ch = np.stack(new_ch, axis=dim)
        x = np.concatenate((x, new_ch), axis=dim)
    elif type(x) == torch.Tensor:
        new_ch = torch.stack(new_ch, dim=dim)
        x = torch.cat((x, new_ch), dim=dim)
    if ret_type is None:
        return x
    elif ret_type == "numpy":
        return to_array(x).astype(np.float32)
    elif ret_type == "torch":
        return to_tensor(x).float()
    raise Exception("unhandled type " + str(type(x)))

