import unittest
import numpy as np

from dodis import (render,features,Normalizer,BindingGraph,nearest_donors,
                   Settler,PurposeGraph,purpose_design,stitch_choices,candidate_edges)

class InstrumentTests(unittest.TestCase):
    def test_inverse_and_identity(self):
        rng=np.random.default_rng(3)
        pieces=rng.normal(0,.01,(64,3,8,8)).astype(np.float32)
        img=render(pieces)
        self.assertEqual(img.shape,(3,32,32))
        self.assertLess(abs(float(np.mean(img))-.47),.1)
        for band in range(4):
            trial=np.zeros_like(pieces)
            trial[band*16+2]=pieces[band*16+2]
            x=render(trial)
            delta=x-np.mean(x)  # shape sanity and nonempty patch
            self.assertGreater(float(np.sum(delta**2)),0.)

    def test_features_are_finite_and_standardized(self):
        rng=np.random.default_rng(4)
        p=rng.normal(0,.08,(28,64,3,8,8)).astype(np.float32)
        v=features(p)
        self.assertEqual(v.shape,(28,64,6))
        n=Normalizer.learn(v)
        self.assertTrue(np.isfinite(n.apply(v)).all())
        self.assertEqual(n.apply(features(p[:3])).shape,(3,64,6))

    def test_hidden_truth_cannot_affect_memory_retrieval(self):
        rng=np.random.default_rng(5)
        bank=rng.normal(0,1,(13,64,3,8,8)).astype(np.float32)
        truth=rng.normal(0,1,(64,3,8,8)).astype(np.float32)
        visible=np.zeros(64,dtype=bool)
        visible[::2]=True
        observed=np.zeros_like(truth)
        observed[visible]=truth[visible]
        idx,dist=nearest_donors(observed,visible,bank,k=4)
        changed=truth.copy()
        changed[~visible]=1e5
        kept=np.zeros_like(changed);kept[visible]=changed[visible]
        idx2,dist2=nearest_donors(kept,visible,bank,k=4)
        np.testing.assert_array_equal(idx,idx2)
        np.testing.assert_array_equal(dist,dist2)
        # Different hidden truth does not change any solver input.
        self.assertEqual(len(idx),4)

    def test_binding_learns_from_cooccurrence_not_class_labels(self):
        rng=np.random.default_rng(7)
        signals=rng.normal(size=300)
        z=rng.normal(size=(300,64,6)).astype(np.float32)*.1
        z[:,0,0]=signals
        z[:,1,0]=signals+.02*rng.normal(size=300)
        graph=BindingGraph.learn(z,max_edges=80)
        self.assertEqual(graph.edges.shape,(80,2))
        self.assertEqual(graph.matrices.shape,(80,6,6))
        self.assertIn((0,1),set(map(tuple,graph.edges)))
        self.assertTrue(np.isfinite(graph.score(z[0])))
        shuffled=graph.permuted()
        self.assertFalse(np.array_equal(shuffled.matrices,graph.matrices))

    def test_purpose_graph_is_train_only_regression(self):
        rng=np.random.default_rng(9)
        edges=np.asarray(candidate_edges()[:12],dtype=int)
        u=rng.normal(size=(1800,64,6))
        y=2*u[:,5,1]-1.25*u[:,17,0]+.7*u[:,edges[0,0],5]*u[:,edges[0,1],5]
        fit=PurposeGraph.fit(u[:1300],y[:1300],u[1300:],y[1300:],edges,ridge=.1)
        self.assertGreater(fit.val_r2,.98)
        self.assertTrue(np.isfinite(fit.predict(u[1300])))
        self.assertEqual(purpose_design(u[:10],edges).shape,(10,1+64*6+len(edges)))

    def test_potts_coordinate_descent_monotone_observed_fixed(self):
        rng=np.random.default_rng(11)
        bank=rng.normal(size=(21,64,3,8,8)).astype(np.float32)*.02
        visible=np.ones(64,dtype=bool);visible[::2]=False
        truth=rng.normal(size=(64,3,8,8)).astype(np.float32)*.02
        observed=np.zeros_like(truth);observed[visible]=truth[visible]
        picks,dist=nearest_donors(observed,visible,bank,4)
        normal=Normalizer.learn(features(bank))
        graph=BindingGraph.learn(normal.apply(features(bank)),max_edges=30)
        solver=Settler(observed,visible,bank[picks],dist,normal,graph,
                       beta=.05,gamma=.4,lam=0,smooth=.75)
        out=solver.solve(passes=2,seed=41)
        self.assertTrue(out['monotone'])
        self.assertLessEqual(out['energy_after'],out['energy_before']+1e-8)
        np.testing.assert_array_equal(out['pieces'][visible],observed[visible])
        # Verify no target hidden information may be passed as observed.
        leaky=observed.copy();leaky[~visible]=truth[~visible]
        with self.assertRaises(ValueError):
            Settler(leaky,visible,bank[picks],dist,normal,graph)
        
if __name__=='__main__':unittest.main()
