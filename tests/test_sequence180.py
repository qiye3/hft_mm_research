"""120秒GPU历史索引及模型权重体积的直接回归。"""
import pytest
import torch

from mm_research.models.sequence180 import RowSequence180


# 【非核心】未来行变化不影响当前预测，历史缓存不得进入模型权重。
@pytest.mark.parametrize('family', ['mlp', 'gru', 'tcn'])
def test_sequence_visibility_and_weights(family):
    """使用小缓存验证三类网络的时间可见性和反向计算。"""
    values = torch.randn(400, 8, device='cuda')
    model = RowSequence180(values, family, 120).cuda().eval()
    rows = torch.tensor([150, 180], device='cuda')
    before = model(rows)
    values[181:] += 100
    after = model(rows)
    torch.testing.assert_close(before, after)
    assert not any('history_values' in name for name in model.state_dict())
    assert before.shape == (2,)
    model.train()
    model(rows).square().mean().backward()


# 【非核心】直接规范化入选列应与既有完整矩阵规范化后选列逐值一致。
def test_gpu_input_reader_matches_shared_normalization(tmp_path):
    """包括NaN与Inf，保证取消磁盘缓存不会改变值和缺失通道。"""
    import numpy as np
    from mm_research.models.sequence180 import load_gpu_inputs
    from mm_research.multihorizon.common import model_inputs

    source = tmp_path / 'source'
    (source / 'e0').mkdir(parents=True)
    values = np.arange(120, dtype='float32').reshape(20, 6)
    values[3, 1] = np.nan
    values[7, 4] = np.inf
    np.save(source / 'e0/x.npy', values)
    mean = np.arange(6, dtype='float32')
    scale = np.arange(1, 7, dtype='float32')
    selected = np.array([4, 1, 3])
    expected = model_inputs(values, np.arange(20), mean, scale)
    expected = expected[:, np.concatenate([selected, selected + 6])]
    actual = load_gpu_inputs(source, selected, {'mean': mean, 'scale': scale}).cpu().numpy()
    np.testing.assert_array_equal(actual, expected)


# 【非核心】临时内存输入复用同一规范化定义，NPY头与GPU读取需一致。
def test_memory_cache_matches_frozen_inputs(tmp_path):
    """使用小数据验证一次缓存后多模型读取的值和缺失标记。"""
    import json
    import numpy as np
    import yaml
    from mm_research.models.sequence180 import prepare_memory_cache, load_gpu_inputs
    from mm_research.multihorizon.common import model_inputs

    source = tmp_path / 'source'
    fold = source / 'e1/folds/final'
    fold.mkdir(parents=True)
    (source / 'e0').mkdir()
    output = tmp_path / 'output'
    (output / 'inputs').mkdir(parents=True)
    values = np.arange(120, dtype='float32').reshape(20, 6)
    values[3, 1] = np.nan
    mean = np.arange(6, dtype='float32')
    scale = np.arange(1, 7, dtype='float32')
    selected = np.array([4, 1, 3])
    np.save(source / 'e0/x.npy', values)
    np.savez(fold / 'preprocessing.npz', mean=mean, scale=scale)
    (output / 'inputs/selection.json').write_text(json.dumps({'indices': selected.tolist()}))
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({'source': str(source), 'output': str(output)}))
    memory_file = tmp_path / 'normalized.npy'
    prepare_memory_cache(config, memory_file)
    expected = model_inputs(values, np.arange(20), mean, scale)
    expected = expected[:, np.concatenate([selected, selected + 6])]
    np.testing.assert_array_equal(np.load(memory_file), expected)
    actual = load_gpu_inputs(source, selected, {'mean': mean, 'scale': scale}, memory_file)
    np.testing.assert_array_equal(actual.cpu().numpy(), expected)


# 【非核心】三项矩阵乘法必须保持原因果膨胀卷积的前向结果和参数/输入梯度。
@pytest.mark.parametrize('dilation', [1, 16, 64])
def test_causal_dilation_matches_conv1d(dilation):
    """Float64独立Conv1d参考覆盖短、中、长膨胀率。"""
    from mm_research.models.sequence180 import causal_dilated_convolution

    values = torch.randn(2, 4, 120, dtype=torch.float64, requires_grad=True)
    convolution = torch.nn.Conv1d(4, 4, 3, dilation=dilation).double()
    expected = convolution(torch.nn.functional.pad(values, (2 * dilation, 0)))
    actual = causal_dilated_convolution(values, convolution)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    gradient = torch.randn_like(actual)
    actual_gradient = torch.autograd.grad(actual, (values, convolution.weight, convolution.bias),
                                          gradient, retain_graph=True)
    expected_gradient = torch.autograd.grad(expected, (values, convolution.weight, convolution.bias), gradient)
    for actual_values, expected_values in zip(actual_gradient, expected_gradient):
        torch.testing.assert_close(actual_values, expected_values, atol=1e-11, rtol=1e-11)
