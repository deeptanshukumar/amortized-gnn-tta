"""The CUDA allocator must retain evidence of allocations freed before return."""
import pytest
import torch
from amortized_gnn_tta.adaptation.benchmark import sample_process_memory


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Actual CUDA allocator required')
def test_cuda_memory_probe_captures_temporary_allocations():
    pytest.importorskip('psutil')
    size = 4 * 1024 * 1024
    def predict():
        temporary = torch.empty(size, dtype=torch.float32, device='cuda')
        temporary.fill_(1)
        return temporary[:1].clone()
    result = sample_process_memory(predict, device='cuda')
    assert result['cuda_peak_allocated_bytes'] >= result['cuda_start_allocated_bytes'] + size * 4
    assert result['cuda_peak_reserved_bytes'] >= result['cuda_peak_allocated_bytes']
    assert result['sampled_peak_rss_bytes'] >= result['start_rss_bytes'] > 0
