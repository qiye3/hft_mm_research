"""扩展模型需experiments依赖，在GPU环境检查历史索引、网络和实际训练入口。"""

from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
import torch

from mm_research.models.experiments import history_batch, train_neural
from mm_research.models.neural import FactorNetwork


class ModelExperimentTests(unittest.TestCase):
    """人工数据只用于接口验证，实际预测结果来自正式历史样本。"""

    # 【非核心】序列只读取当前及过去，缺失在显式通道中保留。
    def test_history_positions_and_missing_indicator(self):
        """修改未来行不影响已经形成的输入序列。"""
        matrix = np.arange(40, dtype=np.float32).reshape(10, 4)
        matrix[3, 1] = np.nan
        result = history_batch(matrix, np.array([4]), np.array([1, 2]), 3)
        np.testing.assert_equal(result[0, :, :2], [[9, 10], [0, 14], [17, 18]])
        self.assertEqual(result[0, 1, 2], 1)
        matrix[5:] = -10000
        np.testing.assert_array_equal(result, history_batch(matrix, np.array([4]), np.array([1, 2]), 3))

    # 【非核心】六种网络都接收同一历史张量并提供有限梯度。
    def test_network_shapes_and_gradients(self):
        """CPU执行小张量，核对序列长度变化与回归输出维度。"""
        torch.set_num_threads(2)
        for family in ['mlp', 'temporal_linear', 'gru', 'lstm', 'tcn', 'transformer']:
            model = FactorNetwork(family, 12, 16, 16)
            output = model(torch.randn(3, 16, 12))
            self.assertEqual(tuple(output.shape), (3,))
            self.assertTrue(torch.isfinite(output).all())
            output.square().mean().backward()
            self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))

    # 【非核心】通过真实CUDA训练与验证入口保存检查点，再核对元数据与模型列顺序。
    def test_cuda_training_checkpoint(self):
        """四个训练步骤覆盖混合精度、验证、最佳模型保存与重新读取。"""
        random = np.random.default_rng(17)
        matrix = random.normal(size=(1000, 6)).astype(np.float32)
        response = (matrix[:, 0] * .2 + matrix[:, 1] * .1).astype(np.float32)
        targets = np.repeat(response[:, None], 5, axis=1)
        metadata = {'columns': [str(index) for index in range(6)], 'train_ic': [[.1] * 6] * 5,
                    'train_stop': 600, 'validation_start': 600, 'validation_stop': 800,
                    'test_start': 800, 'rows': 1000, 'target_scale_bps': [1.] * 5}
        trial = {'family': 'gru', 'horizon_seconds': 1, 'sequence_seconds': 16, 'hidden': 16,
                 'seed': 17, 'learning_rate': .001, 'steps': 4, 'batch_size': 32}
        config = {'selected_columns': 6, 'validation_samples': 100, 'maximum_trial_seconds': 60, 'validate_every_steps': 2}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            report = train_neural(trial, config, metadata, matrix, targets, output, time.time() + 60)
            self.assertEqual(report['steps_completed'], 4)
            self.assertTrue(np.isfinite(report['validation']['mse_bps2']))
            self.assertTrue((output / 'best.pt').exists())


if __name__ == '__main__':
    unittest.main()
