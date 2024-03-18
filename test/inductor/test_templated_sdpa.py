# Owner(s): ["module: inductor"]
import itertools

import torch

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    TestCase,
)
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA
from torch.testing._internal.triton_utils import requires_cuda
from torch.nn.attention.score_mode import sdpa

import unittest
import torch
from torch.testing._internal.common_utils import TestCase
from torch.nn.attention.score_mode import sdpa
import functools

def create_attention(score_mod):
    return functools.partial(sdpa, score_mod=score_mod)

class TestTemplatedSDPA(TestCase):
    def run_test(self, score_mod):
        sdpa = create_attention(score_mod)
        compiled_sdpa = torch.compile(sdpa)
        q = torch.randn((4, 8, 2048, 64), dtype=torch.float16, device="cuda")
        k = torch.randn((4, 8, 2048, 64), dtype=torch.float16, device="cuda")
        v = torch.randn((4, 8, 2048, 64), dtype=torch.float16, device="cuda")
        ref_out = sdpa(q.to(torch.float64), k.to(torch.float64), v.to(torch.float64))
        compiled_out = compiled_sdpa(q, k, v)
        torch.testing.assert_close(ref_out.to(dtype=torch.float32), compiled_out)

    def test_identity(self):
        score_mod = lambda score, b, h, m, n: score
        self.run_test(score_mod)

    def test_causal_mask(self):
        score_mod = lambda score, b, h, m, n: torch.where(m <= n, score, float("-inf"))
        self.run_test(score_mod)

    def test_rel_bias(self):
        score_mod = lambda score, b, h, m, n: score + (m - n)
        self.run_test(score_mod)

    def test_alibi_bias(self):
        score_mod = lambda score, b, h, m, n: score + (m - n) * h
        self.run_test(score_mod)

    def test_rel_causal(self):
        score_mod = lambda score, b, h, m, n: (score + (m - n)) * torch.where(m <= n, score, float("-inf"))
        self.run_test(score_mod)

    def test_alibi_causal(self):
        score_mod = lambda score, b, h, m, n: (score + (m - n) * h) * torch.where(m <= n, score, float("-inf"))
        self.run_test(score_mod)

if __name__ == '__main__':
    unittest.main()

