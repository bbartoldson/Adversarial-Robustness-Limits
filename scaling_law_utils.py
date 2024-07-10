import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy.optimize import minimize
from scipy.special import huber
import itertools
from time import time
from copy import deepcopy
from functools import partial
from scipy.special import logsumexp
from sympy import symbols, diff, lambdify, init_printing, log, oo
from scipy.optimize import fsolve


def get_data_of_source_and_train_frac(data_source = None, train_fraction_floor=0.5, metric = 'CW_test_loss', optimal_data_only=False):
    
    #metrics: 'CW_test_loss' 'CW_test_acc' ... 
    #data_sources:  'pfgmpp18' 'dg20' 'pfgmpp18' 'vanilla20' 'vanilla5' ... 
    metric_is_loss = 'loss' in metric

    data = load_and_rescale_mixed_fid()

    if optimal_data_only:
        optimal_combos = list(set(list(approach_1('dg20', return_frontier_N_D=True))))
        for i in range(len(optimal_combos)):
            m,d = optimal_combos[i]
            for size in sorted(data.data.unique()):
                if d<=size:
                    optimal_combos[i] = m, size
                    break
        optimal_combos = set(optimal_combos)
            
        data['optimal'] = [(m,d) in optimal_combos or d>100 for m,d in zip(data.model, data.data)]
        print(f'Using only optimal model-data combinations, data was {len(data.config)} observations.')
        data = data.query('optimal')
        print(f'Data is now {len(data.config)} observations.')

    if data_source:
        if isinstance(data_source, str): 
            data = data.query('train_fraction>=@train_fraction_floor').query('data_source==@data_source')
        else:
            temp = data.query('train_fraction>=@train_fraction_floor').query('data_source==@data_source[0]')
            for ds in data_source[1:]:
                temp = pd.concat([temp, data.query('train_fraction>=@train_fraction_floor').query('data_source==@ds')], ignore_index=True)
            data = temp
    print(f'Analyzing data sources {data.data_source.unique()}')
    min_flops = data.query('train_fraction>@train_fraction_floor').flops.min()
    max_flops = data.flops.max()
    range_of_flops_to_consider_log = (np.log10(min_flops)-0.001, np.log10(max_flops)+0.001) # e.g., 1e8 through 1e11
    data['metric'] = data[metric]
    return data, range_of_flops_to_consider_log, metric_is_loss


def load_and_rescale_mixed_fid(drop_v5_875=True):
    #Load data and rescale FIDs corresponding to mixed synthetic-real datasets.
    #By default, the `problematic` `vanilla5_syntheticFraction0.875` will be 
    #dropped from the returned dataframe.
    df = pd.read_csv('AT_scaling_law_data.csv')
    if drop_v5_875:
        df = df[~df.data_source.str.contains("vanilla5_syntheticFraction0.875")]
    rescale_list = ['vanilla5_syntheticFraction0.75',
                    'vanilla7_syntheticFraction0.875',
                    'vanilla7_syntheticFraction0.75',
                    'vanilla20_syntheticFraction0.875', 
                    'vanilla20_syntheticFraction0.75']
    rescale_factors = [0.25, 0.65, 0.7, 1.1, 1.2]
    rescale_dict = dict(zip(rescale_list, rescale_factors))
    data_sources = df['data_source'].unique()
    for data_source in data_sources:
        if data_source in rescale_dict.keys():
            old_FID = df.loc[df['data_source'] == data_source, 'FID'].unique()
            df.loc[df['data_source'] == data_source, 'FID'] *= rescale_dict[data_source]
            new_FID = df.loc[df['data_source'] == data_source, 'FID'].unique()
            #print(f'Rescaled FID for data_source {data_source} from {old_FID} to {new_FID}')
        else:
            old_FID = df.loc[df['data_source'] == data_source, 'FID'].unique()
            #print(f'Keeping same FID for data_source {data_source} of {old_FID}')
    return df

##########################################
### Approach 1 helper functions follow ###
##########################################

def approach_1(data_source = 'pfgmpp18', train_fraction_floor=0.5, metric = 'CW_test_loss', flop_floor = None,
                return_frontier_N_D=False, drop_small_models_on_large_data=False, return_analysis = False):
    data, range_of_flops_to_consider_log, metric_is_loss = get_data_of_source_and_train_frac(data_source, train_fraction_floor=train_fraction_floor, metric = metric)
    if drop_small_models_on_large_data:
        data = data.query('not (params<100e6 & data>10e6 & FID<10)')
    if flop_floor is not None:
        data = data.query('flops>@flop_floor')
    if return_analysis:
        analysis = get_param_data_scaling_coeffs_from_slopes(data, range_of_flops_to_consider_log=range_of_flops_to_consider_log, metric_is_loss=metric_is_loss, return_frontier_N_D=True)
        return analysis
    if return_frontier_N_D:
        analysis = get_param_data_scaling_coeffs_from_slopes(data, range_of_flops_to_consider_log=range_of_flops_to_consider_log, metric_is_loss=metric_is_loss, return_frontier_N_D=True)
        return zip(analysis.models, analysis.data)
    (param_regression_intercept, param_regression_slope,
        data_regression_intercept, data_regression_slope) = get_param_data_scaling_coeffs_from_slopes(data,
            range_of_flops_to_consider_log=range_of_flops_to_consider_log, metric_is_loss=metric_is_loss)
    return param_regression_intercept, param_regression_slope, data_regression_intercept, data_regression_slope

class EnhancedPlotLine: 
    '''
    this class adds extra attributes/methods to a line from a plot.
    basically, use it to interpolate between two flop values on a 
    flop-metric line to get an estimate of what the metric should be
    at a particular flop level.
    '''
    def __init__(self, line):
        self.label = line.get_label()
        self.flops = line.get_xdata()
        self.metric = line.get_ydata()

    def interp(self, query_flops, compare_losses):
        """
        linearly interpolate between flops to get acc at query_flops

        returns:
            1) the interpolated metric at the flops value
        """
        # Find the indices into the sorted array `self.flops` such that, if `query_flops` 
        # was inserted before the index, the order of `self.flops` would be preserved
        idx = np.searchsorted(self.flops, query_flops,'left')
        if idx == 0: # the flops is less than or equal to the min flops of the model
            return self.metric[0]
        # check if the query flops amount is greater than any flop amount we have for this config
        if idx>=len(self.flops): # if so, just select the best value attained by this config (from a lower flops amount)
            if compare_losses:
                return min(self.metric)
            else:
                return max(self.metric)
        # otherwise, return the acc (or loss) corresponding to the query flops using linear interpolation
        alpha = (query_flops-self.flops[idx-1]
                 )/(self.flops[idx]-self.flops[idx-1])
        interpolated_metric = alpha*self.metric[idx] + (1-alpha)*self.metric[idx-1]
        return interpolated_metric

def calculate_amount_of_data_at_flops(query_flops, label, lookup):
    max_flops_in_run = lookup[label]["max_flops_in_run"]
    return min(1, query_flops/max_flops_in_run) * lookup[label]["data"]

class Analysis:
    '''
    use this to extract the "envelope" from a bunch of lines (`objects`)

    compare_losses: whether metric is a loss or not (False if an acc)
    objects: the line data from the plot above
    flop_range: range of flops to analyze
    '''
    def __init__(self, compare_losses: bool, objects, flop_range, lookup):
        (self.bests, self.labels, self.flops, 
         self.models, self.params, self.data) = [],[],[],[],[],[]
        for f in np.logspace(*flop_range, num=1000):
            best_metric = 1e9 if compare_losses else 0
            best_line = None
            for o in objects:
                val = o.interp(f, compare_losses)
                if f<o.flops.min(): # if model has no results for a FLOP level as small as `f`...
                    continue # skip it
                if compare_losses: # compare losses (take smallest)
                    if val<best_metric:
                        best_metric = val
                        best_line = o
                else: # comparing accuracies (take largest)
                    if val>best_metric:
                        best_metric = val
                        best_line = o
            if best_line == None: # we had not results at this query flop value (i.e., `f` was too small)
                continue
            # check if the query flops exceeds the maximum flops of the best model, 
            if f > best_line.flops.max(): # which happens when the best model was trained with fewer than `f` flops.
                continue # we do not include these points, they are due to lack of continuous experimental data
            # do not count the same point twice
            if compare_losses:
                if best_metric > min(self.bests+[best_metric]):
                    continue
            else:
                if best_metric < max(self.bests+[best_metric]):
                    continue
            self.flops.append(f)
            self.bests.append(best_metric)
            self.labels.append(best_line.label)
            self.models.append(lookup[best_line.label]["model"])
            self.params.append(lookup[best_line.label]["params"])
            #self.data.append(lookup[label]["data"]) # this would give the data associated with the experiment
            self.data.append(calculate_amount_of_data_at_flops(f, best_line.label, lookup)) # this would give the data associated with the experiment at the query flops level `f`

def R2(y_pred, y):
    RSS = np.sum((y - y_pred)**2)
    TSS = np.sum((y - np.mean(y))**2)

    return 1 - (RSS / TSS)  

def get_param_data_scaling_coeffs_from_slopes(data, range_of_flops_to_consider_log=None, metric_is_loss=True, return_frontier_N_D=False):
    # information dictionary for each config
    lookup = {d[1].config:{
                    'params':d[1].params,
                    'model':d[1].model, 
                    'data':d[1].data_precise,
                    'max_flops_in_run': d[1].flops
                    } 
                for d in data.groupby('config').max().reset_index().iterrows()}
    plt.ioff()
    for c in data.config.unique():
        plt.plot(data[data.config==c].flops,
                data[data.config==c].metric, label=c)
    plt.gca().legend().set_visible(False)
    plt.xscale('log')
    flop_metric_lines = plt.gca() # here we save the lines!!!
    enhanced_lines = []
    for l in flop_metric_lines.lines:
        enhanced_lines.append(EnhancedPlotLine(l))
    analysis = Analysis(compare_losses=metric_is_loss, objects=enhanced_lines, flop_range=range_of_flops_to_consider_log, lookup=lookup)
    plt.close()
    if return_frontier_N_D:
        return analysis
    param_regression = linregress(np.log10(analysis.flops), np.log10(analysis.params))
    data_regression = linregress(np.log10(analysis.flops), np.log10(analysis.data))
    return param_regression.intercept, param_regression.slope, data_regression.intercept, data_regression.slope



##########################################
### Approach 3 helper functions follow ###
##########################################

def approach_3(data_source = 'pfgmpp18', train_fraction_floor=0.5, metric = 'CW_test_loss', optimal_data_only=False, fix_e=None,
               drop_small_models_on_large_data = False):
    data, range_of_flops_to_consider_log, metric_is_loss = get_data_of_source_and_train_frac(data_source, train_fraction_floor=train_fraction_floor, metric = metric, optimal_data_only=optimal_data_only)
    data = scale_params_by_1M(data)
    if drop_small_models_on_large_data:
        data = data.query('not (params<100e6 & data>10e6 & FID<10)')
    L, N, D, F, FID = get_LNDFlopsFID_at_optimum(data, metric_is_loss=metric_is_loss)
    loss = partial(approach_3_loss, metric_is_loss)
    minimize_result = grid_search_lbfgs_minimize(L, N, D, loss=loss, metric_is_loss=metric_is_loss, fix_e=fix_e)
    A = np.exp(minimize_result.x[0])
    B = np.exp(minimize_result.x[1])
    E = np.exp(minimize_result.x[2])
    alpha = minimize_result.x[3]
    beta = minimize_result.x[4]
    a,b = get_scaling_coeffs(alpha, beta)
    return A, B, E, alpha, beta, a, b

def grid_search_lbfgs_minimize(L, N, D, loss = None, metric_is_loss=True, fix_e=None):
    a_s = [0,1,2,5,10]
    b_s = [0,1,2,5,10]
    alphas = [0,0.1,0.25,0.5,1]
    betas = [0,0.1,0.25,0.5,1]
    e_s = [-1,-.5,0,.5,1]
    if not metric_is_loss:
        a_s = [-5, -4.5, -4, -3.5, -3]
        b_s = [-5, -4.5, -4, -3.5, -3]
    best = 1e10
    start = time()
    bounds = None
    if fix_e:
        bounds = [(-np.infty,np.infty),(-np.infty,np.infty),(fix_e,fix_e),(-np.infty,np.infty),(-np.infty,np.infty)]
    for i, (a, b, e, alpha, beta) in enumerate(itertools.product(a_s, b_s, e_s, alphas, betas)):
        if i%1000==0:
            print(i, (time()-start)/60, best, loss([a,b,e,alpha,beta],L,N,D))
        test = minimize(loss, [a,b,e,alpha,beta], args=(L, N, D), method='L-BFGS-B',
                            jac=None, hess=None, hessp=None, bounds=bounds, 
                            constraints=(), tol=None, callback=None, options=None)
        if test.fun < best:
            best = test.fun
            minimize_result = test
    print(i, (time()-start)/60, best, loss([a,b,e,alpha,beta],L,N,D))
    return minimize_result

def scale_params_by_1M(data):
    d = deepcopy(data)
    d.params = data.params * 1e6
    d.data = data.data * 1e6
    d.data_precise = data.data_precise * 1e6
    return d

def log_sum_exp(x):
    return logsumexp(x, axis=1)

def approach_3_loss(metric_is_loss, x, L, N, D):
    delta = 1e-3 # from chinchilla paper
    a, b, e, alpha, beta = x
    if metric_is_loss:
        observations = np.array([a - alpha*np.log(N), b - beta*np.log(D), [e]*len(N)]).T # n x 3 array
    else:
        observations = np.array([a + alpha*np.log(N), b + beta*np.log(D), [e]*len(N)]).T # n x 3 array
    error = log_sum_exp(observations) - np.log(L)
    assert error.shape == L.shape
    #return np.mean(error**2)
    return np.mean(huber(delta, error))

def get_LNDFlopsFID_at_optimum(data, metric_is_loss=True, data_source=None, metric=None):
    temp = data.copy()
    data = None
    if metric is not None:
        temp['metric'] = temp[metric]
    if data_source is not None:
        temp = temp.query('data_source==@data_source')
    if metric_is_loss:
        L, N, D, F, FID, tf = temp.loc[temp.groupby('config')['metric'].idxmin()][['metric', 'params', 'data_precise', 'flops', 'FID', 'train_fraction']].values.T
    else:
        L, N, D, F, FID, tf = temp.loc[temp.groupby('config')['metric'].idxmax()][['metric', 'params', 'data_precise', 'flops', 'FID', 'train_fraction']].values.T
    D*=tf
    return L, N, D, F, FID

def get_scaling_coeffs(alpha, beta):
    a = beta/(alpha+beta)
    b = alpha/(alpha+beta)
    return a, b

def get_scale_for_FLOPS_eq_NDscale(data):
    ''''
    scale = 880 # for FLOPs = FWD_FLOPS*3
    scale = 10720 # for FLOPs = FWD_FLOPS*37
    '''
    scale = 7822 # for FLOPs = FWD_FLOPS*27
    (data.flops / (data.params*data.data_precise*data.train_fraction*scale)).mean()
    assert (data.flops / (data.params*data.data_precise*data.train_fraction*scale)).max() < 1.01, (data.flops / (data.params*data.data_precise*data.train_fraction*scale)).max()
    assert (data.flops / (data.params*data.data_precise*data.train_fraction*scale)).min() > 0.99, (data.flops / (data.params*data.data_precise*data.train_fraction*scale)).min()
    return scale

def N_star(C, A, B, alpha, beta, metric_is_loss=True, scale=None):
    if not metric_is_loss:
        G = (beta*B/(alpha*A))**(1/(beta+alpha))
    else:
        G = (alpha*A/(beta*B))**(1/(beta+alpha))
    return G*(C/scale)**(beta/(beta+alpha))

def D_star(C, A, B, alpha, beta, metric_is_loss=True, scale=None):
    if not metric_is_loss:
        G = (beta*B/(alpha*A))**(1/(beta+alpha))
    else:
        G = (alpha*A/(beta*B))**(1/(beta+alpha))
    return G**-1*(C/scale)**(alpha/(beta+alpha))

def L_hat(N, D, A, B, E, alpha, beta, metric_is_loss=True):
    """ 
    Approximates loss given N parameters and D dataset size (in images),
    similar to Chinchilla paper.
    """
    if not metric_is_loss:
        return A * (N ** alpha) + B * (D ** beta) + E
    return A / (N ** alpha) + B / (D ** beta) + E

def nf_to_d(n, f, scale=None):
    return f/(n*scale)

def df_to_n(d, f, scale=None):
    return f/(d*scale)



###############################################################
### Approach 3 with FID (DG20 base) helper functions follow ###
###############################################################

class chinchilla_approach_3_FID_with_base():
    
    def __init__(self, train_fraction_floor=0.5, metric = 'CW_test_loss', data_source=None, 
                drop_small_models_on_large_data=False, base_params = None):
        self.A, self.B, self.E, self.alpha, self.beta, self.FID = base_params
        data, range_of_flops_to_consider_log, self.metric_is_loss = get_data_of_source_and_train_frac(data_source=data_source,
                                                                                                train_fraction_floor=train_fraction_floor, metric = metric)
        self.data = scale_params_by_1M(data)    
        if drop_small_models_on_large_data:
            self.data = self.data.query('not (params<100e6 & data>10e6 & FID<10)')

    def fit(self):
        L, N, D, F, FID = get_LNDFlopsFID_at_optimum(self.data, metric_is_loss=self.metric_is_loss)
        loss = partial(self.approach_3_FID_with_base_loss, self.metric_is_loss)
        minimize_result = self.grid_search_lbfgs_minimize_FID_with_base(L, N, D, FID, loss=loss, metric_is_loss=self.metric_is_loss)

        self.FID_e = minimize_result.x[0]
        self.FID_b = minimize_result.x[1]

        return self.A, self.B, self.E, self.alpha, self.beta, self.FID_e, self.FID_b, get_scaling_coeffs(self.alpha, self.beta)

    @property
    def arg_min_dict(self):
        d = dict(
                A = self.A,
                B = self.B,
                E = self.E,
                alpha = self.alpha,
                beta = self.beta,
                FID_e = self.FID_e,
                FID_b = self.FID_b,
        )
        return d

    def FID_with_base_modifier(self, param, FID_param, FID, log_first=False):
        modification = np.log(1+FID)*FID_param - np.log(1+self.FID)*FID_param
        if log_first:
            param = np.log(param) + modification
            return np.exp(param)
        return param + modification

    def approach_3_FID_with_base_loss(self, metric_is_loss, x, L, N, D, FID):
        delta = 1e-3 # from chinchilla paper
        a, b, e, alpha, beta  = [np.log(x) for x in (self.A, self.B, self.E)] + [self.alpha, self.beta]
        FID_e, FID_b = x
        b = self.FID_with_base_modifier(b, FID_b, FID)
        e = self.FID_with_base_modifier(e, FID_e, FID)
        if metric_is_loss:
            observations = np.array([a - alpha*np.log(N), b - beta*np.log(D), e]).T # n x 3 array
        else:
            observations = np.array([a + alpha*np.log(N), b + beta*np.log(D), e]).T # n x 3 array
        error = log_sum_exp(observations) - np.log(L)
        assert error.shape == L.shape
        return np.mean(huber(delta, error))

    def grid_search_lbfgs_minimize_FID_with_base(self, L, N, D, FID, loss = None, metric_is_loss=True):
        FID_bs = np.array([-.3, -.15, .15, .3])
        FID_es = np.array([0.01, 0.1, 0.2])
        best = 1e10
        start = time()
        for i, (FID_e, FID_b) in enumerate(itertools.product(FID_es, FID_bs)):
            if i%1000==0:
                print(i, (time()-start)/60, best)
            test = minimize(loss, [FID_e, FID_b], args=(L, N, D, FID), method='L-BFGS-B',
                                jac=None, hess=None, hessp=None, 
                                bounds= None,
                                constraints=(), tol=None, callback=None, options=None)
            if test.fun < best:
                best = test.fun
                minimize_result = test
        print(i, (time()-start)/60, best)
        return minimize_result
    
    def L_hat(self, N, D, FID):
        """ 
        """
        l_hat = L_hat(N, D,
                        self.A, 
                        self.FID_with_base_modifier(self.B, self.FID_b, FID, log_first=True),
                        self.FID_with_base_modifier(self.E, self.FID_e, FID, log_first=True),
                        self.alpha, 
                        self.beta
        )
        return l_hat

    def get_D_star_N_star(self, FLOPs, data_source, override_FID=None):
        data, _, _ = get_data_of_source_and_train_frac(data_source=data_source)
        data = scale_params_by_1M(data)    
        scale = get_scale_for_FLOPS_eq_NDscale(data)
        FID = data.FID.unique().item() if override_FID is None else override_FID
        n_star=N_star(FLOPs, self.A, self.FID_with_base_modifier(self.B, self.FID_b, FID, log_first=True),
                                        self.alpha, self.beta, scale=scale)
        d_star=D_star(FLOPs, self.A, self.FID_with_base_modifier(self.B, self.FID_b, FID, log_first=True),
                                        self.alpha, self.beta, scale=scale)
        assert all(abs(d_star-nf_to_d(n_star, FLOPs, scale))<10), (d_star, nf_to_d(n_star, FLOPs, scale))
        pred_l_star = self.L_hat(n_star, d_star, np.array([FID]))
        d = dict(data=data_source, FLOPs=FLOPs, FID=FID,
                a=self.beta/(self.beta+self.alpha), b = self.alpha/(self.beta+self.alpha),
                N_star=n_star,
                D_star=d_star,
                l_hat_star=pred_l_star
                )
        return d



#####################################################################
#### Equation 1.5 + Chinchilla form at FID=0 helper class follows ###
#####################################################################

class Equation_1_5_That_Is_Chinchilla_At_FID0:
    '''
    Differs from related version below in that it uses delta=1e-2 and 1+FID everywhere instead of FID
    in some places
    '''
    def __init__(self, train_fraction_floor=0.5, metric = 'CW_test_loss',
                 data_source=None, model=None, drop_small_models_on_large_data=False,
                 FID_modifies_E=False, delta = 1e-3):
        data, range_of_flops_to_consider_log, metric_is_loss = get_data_of_source_and_train_frac(
                                                                        data_source=data_source,
                                                                        train_fraction_floor=train_fraction_floor, 
                                                                        metric = metric)
        data = scale_params_by_1M(data)
        self.drop_small_models_on_large_data = drop_small_models_on_large_data
        if self.drop_small_models_on_large_data:
            data = data.query('not (params<100e6 & data>10e6 & FID<10)')
        self.data = data
        self.fit_N = True
        self.loss = partial(self.equation_1_5_FID_loss, metric_is_loss)
        self.l_hat = self.equation_1_5_L_hat
        self.metric_is_loss = metric_is_loss
        self.FID_modifies_E = FID_modifies_E
        self.delta = delta

    def fit(self):
        L, N, D, F, FID = get_LNDFlopsFID_at_optimum(self.data, metric_is_loss=self.metric_is_loss)
        minimize_result = self.grid_search_lbfgs_minimize_equation_1_5(L, N, D, FID,
                                                                  loss=self.loss,
                                                                    metric_is_loss=self.metric_is_loss)
        self.arg_min = minimize_result.x
        return self.arg_min

    @property
    def arg_min_dict(self):
        d = dict(
                A = self.arg_min[0],
                B = self.arg_min[1],
                E = self.arg_min[2],
                Q = self.arg_min[3],
                alpha = self.arg_min[4],
                beta = self.arg_min[5],
                kappa = self.arg_min[6],
        )
        if not self.FID_modifies_E:
            # only return  A, B, E, Q, alpha, beta, kappa
            return d
        d['E_FID'] = self.arg_min[7]
        return d

    def equation_1_5_FID_loss(self, metric_is_loss, x, L, N, D, FID):
        delta = self.delta # from chinchilla paper
        observations = self.l_hat(N, D, FID, learnable_parameters=x,
                                    metric_is_loss=metric_is_loss, return_log=True)
        error = observations - np.log(L)
        assert error.shape == L.shape
        return np.mean(huber(delta, error))

    def equation_1_5_L_hat(self, N, D, FID, learnable_parameters=None, metric_is_loss=True, return_log=False):
        """ 
        Approximates loss given N parameters and D dataset size (in images),
        similar to Neural Scaling Laws Eq 1.5 but couples FID and data.

        Quality = 1/FID
        """
        if not self.FID_modifies_E:
            if learnable_parameters is not None:
                A, B, E, Q, alpha, beta, kappa = learnable_parameters
            else:
                A, B, E, Q, alpha, beta, kappa = self.arg_min
            E = [E]*len(FID)
        else:
            if learnable_parameters is not None:
                A, B, E, Q, alpha, beta, kappa, e_FID = learnable_parameters
            else:
                A, B, E, Q, alpha, beta, kappa, e_FID = self.arg_min
            E = E + np.log(1+FID)*e_FID
        A = np.array([A])
        observations = log_sum_exp(np.array([ 
                                            beta * np.log((B/D) + (Q*FID)**(kappa/beta)),
                                            np.log(E),
                                            np.log(A) - alpha * np.log(N)
                                            ]).T)
        if return_log:
            return observations
        return np.exp(observations)

    def get_grid_search_values(self):
        

        if self.drop_small_models_on_large_data and self.FID_modifies_E:
            a_s = np.array([5,6,7]) 
            b_s = np.array([6.5,7,7.5]) * 1e3 
            e_s = [.6, .5]
            q_s = [0.01, 0.5]
            alphas = [0.1, .2, .3]
            betas = [0.1, .2, .3]
            kappas = np.array([.8,.6]) 
            e_FIDs = [.01]

            return a_s, b_s, e_s, q_s, alphas, betas, kappas, e_FIDs
        

    def grid_search_lbfgs_minimize_equation_1_5(self, L, N, D, FID, loss = None, metric_is_loss=True):

        initial_values_for_learnable_parameters = self.get_grid_search_values()
        best = 1e10
        start = time()
        for i, learnable_parameters in enumerate(itertools.product(*initial_values_for_learnable_parameters)):
            if i%1000==0:
                print(i, (time()-start)/60, best)
            test = minimize(loss, learnable_parameters, args=(L, N, D, FID), method='L-BFGS-B',
                                jac=None, hess=None, hessp=None, 
                                bounds= None,
                                constraints=(), tol=None, callback=None, options=None)
            if test.fun < best:
                best = test.fun
                minimize_result = test
        print(i, (time()-start)/60, best)
        return minimize_result
    
    def get_symbols(self):
        return symbols('A B E Q alpha beta kappa epsilon Quality C N D scale', real=True)
    
    def get_expr(self, N_estimate=True):
        _A, _B, _E, _Q, _alpha, _beta, _kappa, _epsilon, _Quality, _C, _N, _D, _scale = self.get_symbols()
        if N_estimate:
            _N = _C/(_scale*_D)
        else:
            _D = _C/(_scale*_N)
        return ((_B/_D) + (_Q/_Quality)**(_kappa/_beta))**_beta + _E +_epsilon*log(1+_Quality**-1) + _A/(_N**_alpha)

    def get_D_star_N_star(self, FLOPs, data_source, override_FID=None):

        FID=override_FID
        if FID is None:
            _, N, D, F, FID = get_LNDFlopsFID_at_optimum(self.data.query('model=="wrn-82-16-swish"').query('data_source==@data_source'),
                                                        metric_is_loss=self.metric_is_loss)
            FID = FID[0]

        if FID == 0:
            Quality = oo
        else:
            Quality = (FID)**-1
        scale = get_scale_for_FLOPS_eq_NDscale(self.data)
        _A, _B, _E, _Q, _alpha, _beta, _kappa, _epsilon, _Quality, _C, _N, _D, _scale = self.get_symbols()
        expr = self.get_expr()

        dLdD = diff(expr, _D)

        A, B, E, Q, alpha, beta, kappa, epsilon = self.arg_min
        dLdD_simple = dLdD.subs(_A,A).subs(_alpha,alpha).subs(_Quality,Quality).subs(_Q,Q).subs(
                        _beta,beta).subs(_kappa,kappa).subs(_C,FLOPs).subs(_B,B).subs(_scale,scale).subs(_epsilon,epsilon)
        
        dLdD_lambda = lambdify([_D], dLdD_simple)

        def func(D):
            return dLdD_lambda(D)
        
        pred_d_star = fsolve(func, [10], xtol=1e-20)[0]
        N = FLOPs/(scale*pred_d_star)
        pred_l_star = self.l_hat(N, pred_d_star, np.array([FID]), self.arg_min).item()

        return dict(data_source=data_source, D_star=pred_d_star, N_star=N, L_star=pred_l_star, FLOPs=FLOPs, FID=FID)

    def get_N_star_D_star(self, FLOPs, data_source, override_FID=None):

        _, N, D, F, FID = get_LNDFlopsFID_at_optimum(self.data.query('model=="wrn-82-16-swish"').query('data_source==@data_source'),
                                                    metric_is_loss=self.metric_is_loss)
        FID = FID[0]

        if override_FID is not None:
            FID=override_FID

        if FID == 0:
            Quality = oo
        else:
            Quality = (FID)**-1
        scale = get_scale_for_FLOPS_eq_NDscale(self.data)
        _A, _B, _E, _Q, _alpha, _beta, _kappa, _epsilon, _Quality, _C, _N, _D, _scale = self.get_symbols()
        expr = self.get_expr(N_estimate=False)

        dLdN = diff(expr, _N)

        A, B, E, Q, alpha, beta, kappa, epsilon = self.arg_min
        dLdN_simple = dLdN.subs(_A,A).subs(_alpha,alpha).subs(_Quality,Quality).subs(_Q,Q).subs(
                        _beta,beta).subs(_kappa,kappa).subs(_C,FLOPs).subs(_B,B).subs(_scale,scale).subs(_epsilon,epsilon)
        
        dLdN_lambda = lambdify([_N], dLdN_simple)

        def func(N):
            return dLdN_lambda(N)
        
        pred_n_star = fsolve(func, [10], xtol=1e-20)[0]
        D = FLOPs/(scale*pred_n_star)
        pred_l_star = self.l_hat(pred_n_star, D, np.array([FID]), self.arg_min).item()

        return dict(data_source=data_source, N_star=pred_n_star, D_star=D, L_star=pred_l_star, FLOPs=FLOPs, FID=FID)