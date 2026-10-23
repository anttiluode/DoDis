"""DoDis v2 visible-only local address, shifted donor and paired result checks.

These tests run on NumPy only. The original 12-bird v1 runner and receipt
remain unchanged; the real 100-bird GPU outcome is deliberately not simulated.
"""
import tempfile
import unittest
from pathlib import Path
import numpy as np

from dodis import features,Normalizer,Settler
from local_memory import LocalPatchMemory,context_for_piece
from run_v2 import paired_stats,atomic_json,MODES


class LocalMemoryTests(unittest.TestCase):
    def bank(self):
        rng=np.random.default_rng(45)
        bank=rng.normal(0,.016,size=(20,64,3,8,8)).astype(np.float32)
        # One distinctive source is centered at tile 10 (r2,c2), while the
        # missing target is at tile 5 (r1,c1): the retrieval MUST permit shift.
        bank[3,10]=.123
        bank[3,11]=.31
        bank[3,48+10]=.08
        return bank

    def observed(self):
        obs=np.zeros((64,3,8,8),dtype=np.float32)
        vis=np.zeros(64,dtype=bool)
        vis[6]=True;obs[6]=.31  # right neighbor of target tile 5
        vis[48+5]=True;obs[48+5]=.08  # same tile, much coarser octave
        return obs,vis

    def test_shifted_local_core_address_recovers_real_patch(self):
        memory=LocalPatchMemory(self.bank(),train_ids=100+np.arange(20))
        observed,visible=self.observed()
        retrieval=memory.retrieve(observed,visible,k=4)
        self.assertEqual(retrieval.candidates.shape,(4,64,3,8,8))
        best=retrieval.provenance[5][0]
        self.assertEqual(best['train_image_id'],103)
        self.assertEqual(best['source_tile'],10)
        self.assertEqual(best['target_tile'],5)
        self.assertEqual(best['band'],0)
        self.assertGreaterEqual(best['visible_context_count'],2)
        np.testing.assert_allclose(retrieval.candidates[0,5],.123,atol=1e-6)
        self.assertTrue(np.isfinite(retrieval.distances).all())

    def test_hidden_target_value_is_not_consulted(self):
        mem=LocalPatchMemory(self.bank())
        observed,vis=self.observed()
        a=mem.retrieve(observed,vis,4)
        # Unobserved target can take ANY true value: it never enters retrieval.
        truth1=observed.copy();truth1[5]=14.1
        truth2=observed.copy();truth2[5]=-3.2
        obs1=np.zeros_like(truth1);obs1[vis]=truth1[vis]
        obs2=np.zeros_like(truth2);obs2[vis]=truth2[vis]
        b=mem.retrieve(obs1,vis,4)
        c=mem.retrieve(obs2,vis,4)
        np.testing.assert_array_equal(a.candidates,b.candidates)
        np.testing.assert_array_equal(b.distances,c.distances)
        np.testing.assert_array_equal(a.candidates,c.candidates)
        # A mistakenly supplied hidden target must be rejected.
        with self.assertRaises(ValueError):
            mem.retrieve(truth1,vis,4)

    def test_absolute_position_is_not_an_address_requirement(self):
        # The target at (1,1) has same relative context as the donor at (2,2).
        observed,visible=self.observed()
        ctx=context_for_piece(5,visible)
        self.assertIn((0,0,1,1.),ctx)
        self.assertIn((3,0,0,1.7),ctx)
        self.assertNotIn((0,0,0,1.),ctx)

    def test_independent_penalties_per_hidden_piece_and_old_mode_unchanged(self):
        rng=np.random.default_rng(5)
        candidates=rng.normal(size=(4,64,3,8,8)).astype(np.float32)*.01
        vis=np.ones(64,dtype=bool)
        vis[[5,9]]=False
        obs=np.zeros((64,3,8,8),dtype=np.float32)
        norm=Normalizer.learn(features(candidates))
        d=np.ones((4,64),np.float32)*5
        d[:,5]=[8,7,0,6]   # piece 5 chooses slot 2
        d[:,9]=[9,0,7,6]   # piece 9 chooses slot 1
        solver=Settler(obs,vis,candidates,d,norm,beta=1.,gamma=0.,smooth=0.)
        out=solver.solve(passes=3,seed=91)
        self.assertTrue(out['monotone'])
        self.assertEqual(int(out['choice'][5]),2)
        self.assertEqual(int(out['choice'][9]),1)
        np.testing.assert_array_equal(out['pieces'][vis],obs[vis])
        # The old scalar-per-donor input is still a supported alternative.
        old=Settler(obs,vis,candidates,np.array([0.,1.,2.,3.]),norm,beta=1.,gamma=0.,smooth=0.)
        self.assertEqual(old.penalty.ndim,1)
        self.assertEqual(solver.penalty.ndim,2)

    def test_explicit_mean_baseline_and_pairwise_stats(self):
        self.assertIn('masked_mean',MODES)
        a=np.arange(100,dtype=float)/100
        b=a+.002
        stat=paired_stats(a,b,B=2000)
        self.assertEqual(stat['n'],100)
        self.assertAlmostEqual(stat['mean_difference_a_minus_b'],-.002,places=8)
        self.assertLess(stat['ci95_bootstrap'][1],0)
        self.assertLess(stat['paired_signflip_p_two_sided'],.05)
        with tempfile.TemporaryDirectory() as d:
            dst=Path(d)/'first.json'
            atomic_json(dst,{'status':'real_receipt_only'})
            self.assertIn('real_receipt_only',dst.read_text())

if __name__=='__main__':
    unittest.main()
