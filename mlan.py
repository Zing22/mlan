import numpy as np
import cvxopt as co
from tqdm import tqdm, trange
co.solvers.options['show_progress'] = False

################################
#
# Author: Zing22
# Github: github.com/Zing22/mlan
# Update Date: 2018/27/9
#
#################################

class MLAN:
    def __init__(self, options, X, Y):
        ## X: A list of node sets, len(X) = v. 
        ##    Each Xi is An (n*k) matrix, means n nodes with k features
        ## Y: Semi-supervised info

        Y_sum = np.sum(Y, axis=1)
        
        ## reorder X, Y
        # labeled and unlabeled nodes indexes
        self.labeled   = [i for i, val in enumerate(Y_sum) if val == 1]
        self.unlabeled = [i for i, val in enumerate(Y_sum) if val == 0]
        
        # reorder Y to ensure labeled nodes arranged at first
        self.Y = self.reorder(Y)
        self.X = []
        for Xi in X:
            self.X.append(self.reorder(Xi))
        
        self.v = len(self.X)
        self.n, self.c = Y.shape
        self.options = options
        
        self.dist = [] # nodes distance matrices list
        for Xv in self.X:
            self.dist.append(self.calcDist(Xv))
    
    def run(self):
        lambda_        = self.options['lambda_']
        k              = self.options['k']
        max_loop_times = self.options['max_loop_times']
        
        n = self.n
        labeled_count = len(self.labeled)
        
        ## init F, the prediction
        F = self.Y.copy()
        
        ## init w
        w = np.ones((self.v, 1)) / self.v
        
        ## calc some useful dist matries
        dist_x = np.zeros((n, n))
        for dist_v in self.dist:
            dist_x += dist_v / self.v
        dist_f = self.calcDist(F)
        
        dist_d = dist_x + lambda_ * dist_f
        
        ## init alpha
        alpha = 0
        sorted_dist_d = np.sort(dist_d, axis=1)
        for i in range(n):
            alpha += k * sorted_dist_d[i, k+1] - np.sum(sorted_dist_d[i, 0:k])
        alpha /= (2 * n)
        ## init alpha ended
        ### note that, alpha won't change any more
        
        ## init S
        S = np.zeros((n, n))
        cvx_P = co.matrix(np.eye(n))
        cvx_G = co.matrix(0.0, (n, n))
        cvx_G[::n + 1] = -1.0
        cvx_A = co.matrix(1.0, (1, n))
        cvx_b = co.matrix(1.0)
        cvx_h = co.matrix(0.0, (n, 1))
        for i in range(n):
            cvx_q = co.matrix(dist_x[i].T / (2 * alpha))
            solve = co.solvers.qp(cvx_P, cvx_q, cvx_G, cvx_h, cvx_A, cvx_b)
            S[i, :] = np.array(solve['x']).ravel()
        ## init S ended
        
        for t in trange(max_loop_times, ascii=True, leave=False):
            S_old = S.copy()
            ## update w
            for i in range(self.v):
                w[i, 0] = 1 / (2 / np.sqrt(np.sum(np.dot(dist_x[i], S))))
            w = w / np.sum(w)
            ## update w ended
            
            ## update F
            S_mean = (S.T + S) / 2
            D_S = np.diag(np.sum(S_mean, axis=0))
            L_S = D_S - S_mean
            
            F_u = - np.matmul(
                        np.matmul(
                            np.linalg.inv(L_S[labeled_count:, labeled_count:]),
                            L_S[labeled_count:, :labeled_count]
                        ),
                        self.Y[:labeled_count]
                    )
            F[labeled_count:] = F_u
            ## update F ended
            
            ## update S
            dist_f = self.calcDist(F)
            dist_x = np.zeros((n, n))
            for i, dist_v in enumerate(self.dist):
                dist_x += dist_v * w[i]
            dist_d = (dist_x + lambda_ * dist_f) / (2 * alpha)
            
            for i in range(n):
                cvx_q = co.matrix(dist_d[i].T)
                solve = co.solvers.qp(cvx_P, cvx_q, cvx_G, cvx_h, cvx_A, cvx_b)
                S[i, :] = np.array(solve['x']).ravel()
                
            ## update S ended
            
            if (np.linalg.norm(S-S_old) / np.linalg.norm(S_old) < 1e-5):
                break
        
        ## re-construct F
        F_result = self.reorder(F, reconstruct=True)
        
        ## return
        return F_result
            
    # calculate distance between nodes' pair.
    def calcDist(self, X):
        n = X.shape[0]
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist[i,j] = dist[j,i] = np.linalg.norm(X[i]-X[j])
        return dist
    
    # re-order nodes in Y according to labeled or not.
    def reorder(self, Xi, reconstruct=False):
        new_X = np.zeros(Xi.shape)
        for i, nidx in enumerate(self.labeled + self.unlabeled):
            if reconstruct:
                new_X[nidx] = Xi[i]
            else:
                new_X[i] = Xi[nidx]
        return new_X


if __name__ == '__main__':
    import random
    n = 100
    v = 5
    classes_count = 4

    # random generate nodes
    X = []
    for i in range(v):
        X.append(np.random.random((n, 20)))

    # random labeled nodes
    Y = np.zeros((n, classes_count)) # into 4 classes
    for i in range(n):
        if random.random() < 0.2:
            Y[i, random.randint(0, classes_count-1)] = 1
    
    # options
    options = {
        "lambda_": 12,
        "k": 3,
        "max_loop_times": 20
    }

    mlan_obj = MLAN(options, X, Y)
    f = mlan_obj.run()
    print("Classes prediction:\n", np.argmax(f, axis=1))
