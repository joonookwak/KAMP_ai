import unittest
import numpy as np
import pandas as pd
from models import MLP, AR, ARRelation
from run_experiment import windows, point_metrics, normal_clip_metrics, score_model
from ad_models import PCAReconstruction, LOF, IsolationForest


class ProtocolTests(unittest.TestCase):
    def test_mlp_seed_controls_initialization_and_training(self):
        rng=np.random.default_rng(11);x=rng.normal(size=(64,6));y=rng.normal(size=(64,3))
        a=MLP(seed=0,hidden=4,max_epochs=2);b=MLP(seed=0,hidden=4,max_epochs=2)
        c=MLP(seed=1,hidden=4,max_epochs=2)
        for m in [a,b,c]:m.fit(x[:48],y[:48],x[48:],y[48:])
        np.testing.assert_array_equal(a.predict(x),b.predict(x))
        self.assertGreater(np.max(np.abs(a.predict(x)-c.predict(x))),1e-4)

    def test_isolation_forest_seed_reaches_estimator(self):
        rng=np.random.default_rng(12);x=rng.normal(size=(64,6));y=rng.normal(size=(64,3))
        a=IsolationForest(seed=0);b=IsolationForest(seed=1)
        a.fit(x,y,x,y);b.fit(x,y,x,y)
        self.assertEqual(a.estimator.random_state,0);self.assertEqual(b.estimator.random_state,1)
        self.assertGreater(np.max(np.abs(a.score(x,y)-b.score(x,y))),1e-6)

    def test_controlled_windows_have_identical_targets(self):
        d=pd.DataFrame({'split':['train']*12,'clip_id':['a']*6+['b']*6,
                        'AI0_Vibration':range(12),'AI1_Vibration':range(12),'AI2_Current':range(12)})
        a,ay,ai=windows(d,'train',1,np.zeros(3),np.ones(3),minimum_index=4)
        b,by,bi=windows(d,'train',3,np.zeros(3),np.ones(3),minimum_index=4)
        np.testing.assert_array_equal(ai,bi)
        np.testing.assert_array_equal(ay,by)
        self.assertEqual(ai.tolist(),[4,5,10,11])
        np.testing.assert_array_equal(a,b[:,-3:])

    def test_pca_scores_current_relationship_violation(self):
        rng=np.random.default_rng(5);x=rng.normal(size=(240,6));y=x[:,-3:].copy()
        model=PCAReconstruction();model.fit(x[:200],y[:200],x[200:],y[200:])
        original=score_model(model,x[200:],y[200:]);changed=y[200:].copy();changed[:,0]+=10
        altered=score_model(model,x[200:],changed)
        self.assertGreater(altered.mean(),original.mean()+1)

    def test_lof_queries_do_not_change_other_query_scores(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(160,6));y=rng.normal(size=(160,3))
        model=LOF();model.fit(x[:140],y[:140],x[140:],y[140:])
        full=score_model(model,x[140:],y[140:]);single=score_model(model,x[140:141],y[140:141])
        np.testing.assert_allclose(full[0],single[0])

    def test_normal_clip_alarm_counts_any_timestamp(self):
        frame=pd.DataFrame({'source':['normal']*4+['fault'], 'clip_id':['a','a','b','b','c'],
                            'y_pred':[False,True,False,False,True]})
        m=normal_clip_metrics(frame)
        self.assertEqual(m['normal_clips'],2);self.assertEqual(m['normal_alert_clips'],1)
        self.assertEqual(m['normal_clip_alarm_rate'],.5)

    def test_ar_has_no_cross_sensor_inputs(self):
        rng=np.random.default_rng(9)
        x=rng.normal(size=(300,6));y=x[:,-3:]*.8
        model=AR();model.fit(x,y,x,y)
        altered=x.copy();altered[:,2::3]+=100
        np.testing.assert_array_equal(model.predict(x)[:,:2],model.predict(altered)[:,:2])

    def test_relation_recovers_nonlinear_current_effect(self):
        rng=np.random.default_rng(7)
        x=rng.normal(size=(600,6))
        y=np.c_[x[:,-1]**2, x[:,-1]*x[:,-3], .7*x[:,-1]]
        ar=AR();ar.fit(x[:400],y[:400],x[400:],y[400:])
        relation=ARRelation();relation.fit(x[:400],y[:400],x[400:],y[400:])
        base=np.mean((ar.predict(x[400:])-y[400:])**2)
        improved=np.mean((relation.predict(x[400:])-y[400:])**2)
        self.assertLess(improved,base*.1)

    def test_mlp_gradient(self):
        rng=np.random.default_rng(0);m=MLP(hidden=4)
        x=rng.normal(size=(5,6));y=rng.normal(size=(5,3))
        ws=[rng.normal(size=(6,4))*.1,np.zeros(4),rng.normal(size=(4,3))*.1,np.zeros(3)]
        _,gs=m.loss_and_gradient(x,y,ws)
        for j,w in enumerate(ws):
            ix=(0,)*w.ndim;old=w[ix];eps=1e-6
            w[ix]=old+eps;a=m.loss_and_gradient(x,y,ws)[0]
            w[ix]=old-eps;b=m.loss_and_gradient(x,y,ws)[0];w[ix]=old
            self.assertAlmostEqual((a-b)/(2*eps),gs[j][ix],places=6)

    def test_causal_windows_do_not_cross_clips(self):
        d=pd.DataFrame({'split':['train']*8,'clip_id':['a']*4+['b']*4,
                        'AI0_Vibration':range(8),'AI1_Vibration':range(8),'AI2_Current':range(8)})
        x,y,ids=windows(d,'train',2,np.zeros(3),np.ones(3))
        self.assertEqual(ids.tolist(),[2,3,6,7])
        self.assertTrue(np.all(x[:,-3:]<y))
        changed=d.copy();changed.loc[3,'AI0_Vibration']=999
        after,_,_=windows(changed,'train',2,np.zeros(3),np.ones(3))
        np.testing.assert_array_equal(x[0],after[0])

    def test_warmup_fault_counts_as_miss(self):
        d=pd.DataFrame({'y_true':[0,0,1,1], 'y_pred':[False,True,False,True],
                        'status':['warmup','scored','warmup','scored']})
        m=point_metrics(d)
        self.assertEqual([m[k] for k in ['TP','FP','TN','FN']],[1,1,1,1])
        self.assertEqual(m['F1'],.5);self.assertEqual(m['warmup_fault'],1)


if __name__=='__main__':unittest.main()
