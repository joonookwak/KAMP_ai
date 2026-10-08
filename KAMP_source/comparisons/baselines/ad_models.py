"""Normal-only detectors on [past W samples, current sample]. No future inputs."""
import numpy as np


def observed(x, y):
    return np.c_[x, y]


class Detector:
    family = 'distribution'
    score_description = ''

    def __init__(self,seed=0):
        self.seed=seed


class PCAReconstruction(Detector):
    family = 'reconstruction'
    score_description = 'full observed-window PCA reconstruction MSE'

    def fit(self, x, y, vx, vy):
        from sklearn.decomposition import PCA
        self.estimator = PCA(n_components=.9, svd_solver='full').fit(observed(x,y))
        return {'explained_variance_target': .9, 'retained_components': int(self.estimator.n_components_),
                'explained_variance_ratio': self.estimator.explained_variance_ratio_.tolist(),
                'selection': '90% normal training variance; fixed before test'}

    def reconstruct(self, x, y):
        z=observed(x,y)
        return self.estimator.inverse_transform(self.estimator.transform(z))

    def score(self, x, y):
        return np.mean((observed(x,y)-self.reconstruct(x,y))**2,axis=1)


class Mahalanobis(Detector):
    family = 'distance'
    score_description = 'Ledoit-Wolf shrinkage squared Mahalanobis distance'

    def fit(self, x, y, vx, vy):
        from sklearn.covariance import LedoitWolf
        self.estimator=LedoitWolf().fit(observed(x,y))
        return {'covariance': 'LedoitWolf', 'shrinkage': float(self.estimator.shrinkage_)}

    def score(self, x, y):
        return self.estimator.mahalanobis(observed(x,y))


class KNN(Detector):
    family = 'distance'
    score_description = 'distance to 20th normal-training neighbor'

    def fit(self, x, y, vx, vy):
        from sklearn.neighbors import NearestNeighbors
        self.estimator=NearestNeighbors(n_neighbors=20,n_jobs=1).fit(observed(x,y))
        return {'neighbors':20,'selection':'fixed; all training windows used'}

    def score(self, x, y):
        return self.estimator.kneighbors(observed(x,y),return_distance=True)[0][:,-1]


class GMM(Detector):
    family = 'density'
    score_description = 'negative Gaussian mixture log likelihood'

    def fit(self, x, y, vx, vy):
        from sklearn.mixture import GaussianMixture
        candidates=[]
        z=observed(x,y)
        for k in [1,2,4]:
            model=GaussianMixture(n_components=k,covariance_type='full',reg_covar=1e-4,
                                  n_init=2,max_iter=200,random_state=self.seed).fit(z)
            candidates.append((float(model.bic(z)),k,model))
        bic,k,self.estimator=min(candidates,key=lambda v:v[0])
        return {'components':k,'covariance_type':'full','reg_covar':1e-4,'n_init':2,
                'seed':self.seed,'selection':'minimum normal-training BIC',
                'candidates':[{'components':k,'BIC':b} for b,k,_ in candidates],
                'converged':bool(self.estimator.converged_)}

    def score(self, x, y):
        return -self.estimator.score_samples(observed(x,y))


class KDE(Detector):
    family = 'density'
    score_description = 'negative Gaussian kernel log density'

    def fit(self, x, y, vx, vy):
        from sklearn.neighbors import KernelDensity
        candidates=[]
        for bandwidth in [.25,.5,1.,2.]:
            model=KernelDensity(bandwidth=bandwidth,kernel='gaussian').fit(observed(x,y))
            loss=float(-model.score_samples(observed(vx,vy)).mean())
            candidates.append((loss,bandwidth,model))
        loss,bw,self.estimator=min(candidates,key=lambda v:v[0])
        return {'bandwidth':bw,'kernel':'gaussian','selection':'normal validation log likelihood',
                'candidates':[{'bandwidth':b,'validation_negative_log_density':v} for v,b,_ in candidates]}

    def score(self, x, y):
        return -self.estimator.score_samples(observed(x,y))


class LOF(Detector):
    family = 'local_density'
    score_description = 'negative novelty-mode LOF score_samples'

    def fit(self, x, y, vx, vy):
        from sklearn.neighbors import LocalOutlierFactor
        self.estimator=LocalOutlierFactor(n_neighbors=20,novelty=True,n_jobs=1).fit(observed(x,y))
        return {'neighbors':20,'novelty':True,'selection':'fixed; scored data never enter neighborhoods'}

    def score(self, x, y):
        return -self.estimator.score_samples(observed(x,y))


class IsolationForest(Detector):
    family = 'isolation'
    score_description = 'negative IsolationForest score_samples'

    def fit(self, x, y, vx, vy):
        from sklearn.ensemble import IsolationForest as IF
        self.estimator=IF(n_estimators=200,max_samples=256,random_state=self.seed,n_jobs=1).fit(observed(x,y))
        return {'trees':200,'max_samples':256,'seed':self.seed,'selection':'fixed; built-in threshold unused'}

    def score(self, x, y):
        return -self.estimator.score_samples(observed(x,y))


class OneClassSVM(Detector):
    family = 'boundary'
    score_description = 'negative RBF OneClassSVM decision_function'

    def fit(self, x, y, vx, vy):
        from sklearn.svm import OneClassSVM as SVM
        self.estimator=SVM(kernel='rbf',nu=.01,gamma='scale').fit(observed(x,y))
        return {'kernel':'rbf','nu':.01,'gamma':'scale','selection':'fixed; built-in zero threshold unused'}

    def score(self, x, y):
        return -self.estimator.decision_function(observed(x,y))


class RobustMAD(Detector):
    family = 'robust_statistic'
    score_description = 'max absolute current-sensor robust z score; history unused'

    def fit(self, x, y, vx, vy):
        self.center=np.median(y,axis=0)
        self.scale=np.maximum(1.4826*np.median(np.abs(y-self.center),axis=0),1e-6)
        self.weights=[self.center,self.scale]
        return {'input':'current 3 sensors only','normal_center':'median','scale':'1.4826 MAD',
                'warmup':'same W as other methods for controlled comparison'}

    def score(self, x, y):
        return np.max(np.abs((y-self.center)/self.scale),axis=1)


class Autoencoder(Detector):
    family = 'reconstruction'
    score_description = 'full observed-window nonlinear autoencoder reconstruction MSE'

    def fit(self, x, y, vx, vy):
        import torch
        from torch import nn
        torch.set_num_threads(1);torch.manual_seed(self.seed);torch.use_deterministic_algorithms(True)
        rng=np.random.default_rng(self.seed)
        z=observed(x,y);v=observed(vx,vy);dim=z.shape[1]
        bottleneck=max(2,min(8,dim//4))
        self.net=nn.Sequential(nn.Linear(dim,32),nn.Tanh(),nn.Linear(32,bottleneck),nn.Tanh(),
                               nn.Linear(bottleneck,32),nn.Tanh(),nn.Linear(32,dim))
        tz=torch.tensor(z,dtype=torch.float32);tv=torch.tensor(v,dtype=torch.float32)
        opt=torch.optim.Adam(self.net.parameters(),lr=.001,weight_decay=1e-4)
        best_mse,stale=float('inf'),0;history=[]
        for epoch in range(1,151):
            self.net.train();order=rng.permutation(len(z))
            for start in range(0,len(order),128):
                ids=order[start:start+128];opt.zero_grad()
                loss=((self.net(tz[ids])-tz[ids])**2).mean();loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(),1.);opt.step()
            self.net.eval()
            with torch.no_grad():mse=float(((self.net(tv)-tv)**2).mean())
            if not np.isfinite(mse):raise ValueError('Nonfinite AE validation loss')
            history.append({'epoch':epoch,'validation_mse':mse})
            if mse<best_mse-1e-6:
                best_mse,best_epoch,stale=mse,epoch,0
                best={k:v.detach().clone() for k,v in self.net.state_dict().items()}
            else:stale+=1
            if stale>=15:break
        self.net.load_state_dict(best);self.net.eval()
        self.weights=[v.detach().numpy().copy() for v in self.net.state_dict().values()]
        return {'architecture':[dim,32,bottleneck,32,dim],'seed':self.seed,'torch_version':torch.__version__,
                'learning_rate':.001,'weight_decay':1e-4,'batch_size':128,'max_epochs':150,
                'early_stopping_patience':15,'best_epoch':best_epoch,'epochs_run':epoch,
                'state_dict_keys':list(self.net.state_dict()),'history':history}

    def reconstruct(self, x, y):
        import torch
        with torch.no_grad():
            return self.net(torch.tensor(observed(x,y),dtype=torch.float32)).numpy().astype(float)

    def score(self, x, y):
        return np.mean((observed(x,y)-self.reconstruct(x,y))**2,axis=1)


AD_MODELS={'PCA_Recon':PCAReconstruction,'AE_Recon':Autoencoder,'Mahalanobis':Mahalanobis,
           'kNN20':KNN,'GMM':GMM,'KDE':KDE,'LOF20':LOF,'IsolationForest':IsolationForest,
           'OCSVM':OneClassSVM,'RobustMAD':RobustMAD}
