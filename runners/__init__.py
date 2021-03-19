"""
Patch missing operators and missing modules
"""
import paddle


if 'cumprod' not in paddle.__dict__:
    def cumprod(x, axis=None):
        if axis is None:
            x = x.reshape([-1])
            axis = 0
        assert isinstance(axis, int)

        axis_length = x.shape[axis]

        slices = []
        for i in range(len(x.shape)):
            if i == axis or i == len(x.shape) + axis:
                slc = [slice(0, j) for j in range(1, axis_length+1)]
            else:
                slc = [slice(0, x.shape[i]) for _ in range(1, axis_length+1)]
            slices.append(slc)

        def get_slc(index):
            return [slices[i][index] for i in range(len(x.shape))]

        y = paddle.stack([
            paddle.prod(x[tuple(get_slc(i))], axis=axis)
            for i in range(axis_length)
        ], axis)
        if len(x.shape) == 1:
            y = y[:,0]
        
        return y

    paddle.cumprod = cumprod
    paddle.Tensor.cumprod = lambda self, axis=None: cumprod(self, axis)

if 'Subset' not in paddle.io.__dict__:
    class Subset(paddle.io.Dataset):
        def __init__(self, dataset, indices) -> None:
            self.dataset = dataset
            self.indices = indices

        def __getitem__(self, idx):
            return self.dataset[self.indices[idx]]

        def __len__(self):
            return len(self.indices)

    paddle.io.Subset = Subset
