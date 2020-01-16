from ..utils import *
from sklearn.model_selection import KFold
from statsmodels.formula.api import ols
from sklearn.metrics import mean_squared_error

#below are the list of constant string-literals used for cross-validation
mse = 'mse'
mse_train = mse + '_train'
mse_test = mse + '_test'
delta_mse = 'delta_' + mse
mse_train_and_mse_test = mse_train + '__and__' + mse_test
mse_and_delta_mse = mse + '__and__' + delta_mse
rmse = 'rmse'
rmse_train = rmse + '_train'
rmse_test = rmse + '_test'
delta_rmse = 'delta_' + rmse
rmse_train_and_rmse_test = rmse_train + '__and__' + rmse_test
rmse_and_delta_rmse = rmse + '__and__' + delta_rmse
rsquared = 'rsquared'
adjusted_rsquared = 'rsquared_adj'
condition_no = 'condition_no'
pvals = 'pvals'
condition_no_and_adjusted_rsquared = condition_no + '__and__' + adjusted_rsquared
condition_no_and_rmse_and_delta_rmse = condition_no + '__and__' + rmse_and_delta_rmse
condition_no_and_pvals_and_rsq_and_adjrsq_and_rmse_and_delta_rmse = \
    condition_no \
    + '__and__' + pvals \
    + '__and__' + rsquared \
    + '__and__' + adjusted_rsquared \
    + '__and__' + rmse_and_delta_rmse

cv_scoring_methods = [
    mse_train_and_mse_test
    , mse_and_delta_mse
    , rmse_train_and_rmse_test
    , rmse_and_delta_rmse
    , adjusted_rsquared
    , condition_no
    , condition_no_and_adjusted_rsquared
    , condition_no_and_rmse_and_delta_rmse
    , condition_no_and_pvals_and_rsq_and_adjrsq_and_rmse_and_delta_rmse
]

def cv_score(
    X
    , y
    , feat_combo
    , folds=5
    , scoring_method=rmse_train_and_rmse_test):

    if scoring_method not in cv_scoring_methods:
        raise ValueError("Unknown scoring_method: '{}'".format(scoring_method))

    if scoring_method == mse_train_and_mse_test:
        scores_df = pd.DataFrame(columns=[mse_train, mse_test])

    elif scoring_method == mse_and_delta_mse:
        scores_df = pd.DataFrame(columns=[mse, delta_mse])

    elif scoring_method == rmse_train_and_rmse_test:
        scores_df = pd.DataFrame(columns=[rmse_train, rmse_test])

    elif scoring_method == rmse_and_delta_rmse:
        scores_df = pd.DataFrame(columns=[rmse, delta_rmse])

    elif scoring_method == adjusted_rsquared:
        scores_df = pd.DataFrame(columns=[adjusted_rsquared])

    elif scoring_method == condition_no:
        scores_df = pd.DataFrame(columns=[condition_no])

    elif scoring_method == condition_no_and_adjusted_rsquared:
        scores_df = pd.DataFrame(columns=[condition_no, adjusted_rsquared])

    elif scoring_method == condition_no_and_rmse_and_delta_rmse:
        scores_df = pd.DataFrame(columns=[condition_no, rmse, delta_rmse])

    elif scoring_method == condition_no_and_pvals_and_rsq_and_adjrsq_and_rmse_and_delta_rmse:
        scores_df = pd.DataFrame(columns=[condition_no, rsquared, adjusted_rsquared, pvals, rmse, delta_rmse])


    f = y.columns[0] + '~' + "+".join(feat_combo)
    scores = []

    if folds > 1:
        train_test_indices = KFold(n_splits=folds).split(X)
    else:
        X_train, X_test, _, _ = train_test_split(X, y, test_size=0.30, random_state=42)
        train_test_indices = [(X_train.index, X_test.index)]

    for train_index, test_index in train_test_indices:
        X_train, X_test = X.iloc[train_index][feat_combo], X.iloc[test_index][feat_combo]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        data_fin_df = pd.concat([y_train, X_train], axis=1, join='inner').reset_index()
        model_fit_results = ols(formula=f, data=data_fin_df).fit()

        if scoring_method == mse_train_and_mse_test \
            or scoring_method == mse_and_delta_mse \
            or scoring_method == rmse_train_and_rmse_test \
            or scoring_method == rmse_and_delta_rmse \
            or scoring_method == condition_no_and_rmse_and_delta_rmse \
            or scoring_method == condition_no_and_pvals_and_rsq_and_adjrsq_and_rmse_and_delta_rmse:

            y_hat_train = model_fit_results.predict(X_train)
            y_hat_test = model_fit_results.predict(X_test)
            _mse_train = mean_squared_error(y_train, y_hat_train)
            _mse_test = mean_squared_error(y_test, y_hat_test)

            if scoring_method == mse_train_and_mse_test:
                data = [
                    {
                        mse_train: _mse_train
                        , mse_test: _mse_test
                    }
                ]

            elif scoring_method == rmse_train_and_rmse_test:
                data = [
                    {
                        rmse_train: np.sqrt(_mse_train)
                        , rmse_test: np.sqrt(_mse_test)
                    }
                ]

            elif scoring_method == mse_and_delta_mse:
                data = [
                    {
                        mse: _mse_train
                        , delta_mse: abs(_mse_test - _mse_train)
                    }
                ]

            elif scoring_method == rmse_and_delta_rmse:
                _rmse_test = np.sqrt(_mse_test)
                _rmse_train = np.sqrt(_mse_train)
                data = [
                    {
                        rmse: _rmse_train
                        , delta_rmse: abs(_rmse_test - _rmse_train)
                    }
                ]

            elif scoring_method == condition_no_and_rmse_and_delta_rmse:
                _rmse_test = np.sqrt(_mse_test)
                _rmse_train = np.sqrt(_mse_train)
                data = [
                    {
                        condition_no: model_fit_results.condition_number
                        , rmse: _rmse_train
                        , delta_rmse: abs(_rmse_test - _rmse_train)
                    }
                ]

            elif scoring_method == condition_no_and_pvals_and_rsq_and_adjrsq_and_rmse_and_delta_rmse:
                _rmse_test = np.sqrt(_mse_test)
                _rmse_train = np.sqrt(_mse_train)
                data = [
                    {
                        condition_no: model_fit_results.condition_number
                        , rsquared: model_fit_results.rsquared
                        , adjusted_rsquared: model_fit_results.rsquared_adj
                        , pvals: model_fit_results.pvalues
                        , rmse: _rmse_train
                        , delta_rmse: abs(_rmse_test - _rmse_train)
                    }
                ]

            scores_df = scores_df.append(data, ignore_index=True, sort=False)

        elif scoring_method == adjusted_rsquared:
            data = [{adjusted_rsquared: model_fit_results.rsquared_adj}]
            scores_df = scores_df.append(data, ignore_index=True, sort=False)

        elif scoring_method == condition_no:
            data = [{condition_no: model_fit_results.condition_number}]
            scores_df = scores_df.append(data, ignore_index=True, sort=False)

        elif scoring_method == condition_no_and_adjusted_rsquared:
            data = [
                {
                    condition_no: model_fit_results.condition_number
                    , adjusted_rsquared: model_fit_results.rsquared_adj
                }
            ]
            scores_df = scores_df.append(data, ignore_index=True, sort=False)

    # now compute the mean score over all k-folds
    if scoring_method == mse_train_and_mse_test:
        mean_mse_train = scores_df[mse_train].mean()
        mean_mse_test = scores_df[mse_test].mean()
        mean_cv_score = (mean_mse_train, mean_mse_test)

    elif scoring_method == rmse_train_and_rmse_test:
        mean_rmse_train = scores_df[rmse_train].mean()
        mean_rmse_test = scores_df[rmse_test].mean()
        mean_cv_score = (mean_rmse_train, mean_rmse_test)

    elif scoring_method == mse_and_delta_mse:
        mean_mse = scores_df[mse].mean()
        mean_delta_mse = scores_df[delta_mse].mean()
        mean_cv_score = (mean_mse, mean_delta_mse)

    elif scoring_method == rmse_and_delta_rmse:
        mean_rmse = scores_df[rmse].mean()
        mean_delta_rmse = scores_df[delta_rmse].mean()
        mean_cv_score = (mean_rmse, mean_delta_rmse)

    elif scoring_method == adjusted_rsquared:
        mean_adj_rsq = scores_df[adjusted_rsquared].mean()
        mean_cv_score = mean_adj_rsq

    elif scoring_method == condition_no:
        mean_cond_no = scores_df[condition_no].mean()
        mean_cv_score = mean_cond_no
    
    elif scoring_method == condition_no_and_adjusted_rsquared:
        mean_cond_no = scores_df[condition_no].mean()
        mean_adj_rsq = scores_df[adjusted_rsquared].mean()
        mean_cv_score = (mean_cond_no, mean_adj_rsq)

    elif scoring_method == condition_no_and_rmse_and_delta_rmse:
        mean_cond_no = scores_df[condition_no].mean()
        mean_rmse = scores_df[rmse].mean()
        mean_delta_rmse = scores_df[delta_rmse].mean()
        mean_cv_score = (mean_cond_no, mean_rmse, mean_delta_rmse)

    elif scoring_method == condition_no_and_pvals_and_rsq_and_adjrsq_and_rmse_and_delta_rmse:
        mean_cond_no = scores_df[condition_no].mean()
        mean_rsq = scores_df[rsquared].mean()
        mean_adj_rsq = scores_df[adjusted_rsquared].mean()
        pvals_df = pd.DataFrame(columns=feat_combo)
        for idx, row in  scores_df.iterrows():
            list_pvals = list(row[pvals].values[1:])
            pvals_df = pvals_df.append(pd.Series(list_pvals, index=pvals_df.columns), ignore_index=True, sort=False)
        mean_pvals = []
        for idx, _ in enumerate(feat_combo):
            mean_pvals.append(pvals_df.iloc[:, idx].mean())
        mean_rmse = scores_df[rmse].mean()
        mean_delta_rmse = scores_df[delta_rmse].mean()
        mean_cv_score = (mean_cond_no, mean_rsq, mean_adj_rsq, mean_pvals, mean_rmse, mean_delta_rmse)

    return (X_train, X_test, y_train, y_test, mean_cv_score)

#this method simply houses the combinations (n-choose-k) used in cross-validation
def cv_build_feature_combinations(X, reverse=False, upper_bound=2**18, boundary_test=False):
    feat_combos = dict()
    
    r = range(len(X.columns), 0, -1) if reverse else range(1, len(X.columns)+1)  # build up from potentially worst case
        
    n = max(r)
    
    # determine whether or not we will exhause memory!
    len_total_combos = 0
    
    s_n_choose_k = "{} \\choose {}"
    if not boundary_test:
        s_total_combos = "<br><br>Builing all $\\sum_{i=" + str(min(r)) + "}^{" + str(n) + "}{" + s_n_choose_k.format(n, "i") + "}=$"
        for k in r:
            s_total_combos += " ${" + s_n_choose_k.format(n, k) + "}$" + (" +" if k!=n else "")
            len_combos = nCr(n, k)
            len_total_combos += len_combos
        s_total_combos += " $={}$ combinations of feature set: {}...".format(len_total_combos, X.columns)
        display(HTML(s_total_combos))
    else:
        len_total_combos = 2**n - 1
    
    if len_total_combos > upper_bound:
        display(HTML("<h2><font color=\"red\">Building all combinations of {} features would result in cross-validating {} models which will most likely exhaust memory (exceeds upper bound: {})!  Please reduce the size of the feature set and try again!</font></h2>".format(n, len_total_combos, upper_bound)))
        feat_combos = None
    else:
        if not boundary_test:
            for k in r:
                feat_combos_of_length_k = list(combinations(X, k))
                feat_combos[k] = feat_combos_of_length_k
        else:
            display(HTML("<h2><font color=\"green\">Boundary test PASSED!  Building all combinations of {} features will result in cross-validating {} models, which is less than upper bound: {}.</font></h2>".format(n, len_total_combos, upper_bound)))
    
    display(HTML("All done!"))    
    
    return (feat_combos, len_total_combos)

def cv_selection_dp(
    X
    , y
    , folds=5
    , reverse=False
    , smargs=None):

    cols = ['n_features', 'features', condition_no, rsquared, adjusted_rsquared, pvals, rmse, delta_rmse]
    scores_df = pd.DataFrame(columns=cols)

    target_cond_no = None
    if smargs is not None:
        target_cond_no = smargs['cond_no']
    if target_cond_no is None:
        target_cond_no = 1000 #default definition of "non-colinearity" used by statsmodels - see https://www.statsmodels.org/dev/_modules/statsmodels/regression/linear_model.html#RegressionResults.summary

    cv_feat_combo_map, _ = cv_build_feature_combinations(X, reverse=reverse)

    if cv_feat_combo_map is None:
        return
    
    base_feature_set = list(X.columns)
    n = len(base_feature_set)
    
    best_feat_combo = []
    best_score = None

    best_feat_combos = []
    
    for _, list_of_feat_combos in cv_feat_combo_map.items():
        n_choose_k = len(list_of_feat_combos)
        k = len(list_of_feat_combos[0])
        depth = k-1
        s_n_choose_k = "{} \\choose {}"
        display(
            HTML(
                "<p><br><br>Cross-validating ${}={}$ combinations of {} features (out of {}) over {} folds using score <b>{}</b> and target cond. no = {}...".format(
                    "{" + s_n_choose_k.format(n, k) + "}"
                    , n_choose_k
                    , k
                    , n
                    , folds
                    , condition_no_and_pvals_and_rsq_and_adjrsq_and_rmse_and_delta_rmse
                    , target_cond_no
                )
            )
        )

        n_discarded = 0
        n_met_constraints = 0

        for feat_combo in list_of_feat_combos:           
            feat_combo = list(feat_combo)

            closest_prior_depth = min(len(best_feat_combos)-1, depth-1)
            if depth > 0 and closest_prior_depth >= 0:
                last_best_feat_combo = best_feat_combos[closest_prior_depth]
                last_best_feat_combo_in_current_feat_combo = set(last_best_feat_combo).issubset(set(feat_combo))
                if last_best_feat_combo_in_current_feat_combo:
                    #print("depth is {}, best_feat_combos[last_saved_depth]: {}".format(depth, best_feat_combos[last_saved_depth]))
                    #print("feat_combo: {}".format(feat_combo))
                    #print("best_feat_combos[last_saved_depth] in feat_combo: {}".format(last_best_feat_combo_in_current_feat_combo))
                    pass
                else:
                    n_discarded += 1
                    #display(HTML("DISCARDED feature-combo {} since it is not based on last best feature-combo {}; discarded so far: {}".format(feat_combo, last_best_feat_combo, n_discarded)))
                    continue

            _, _, _, _, score = cv_score(
                X
                , y
                , feat_combo
                , folds
                , condition_no_and_pvals_and_rsq_and_adjrsq_and_rmse_and_delta_rmse
            )

            # now determine if this score is best
            is_in_conf_interval = False not in [True if pval >= 0.0 and pval <= 0.05 else False for pval in score[3]]
            is_non_colinear = score[0] <= target_cond_no
            if is_non_colinear and is_in_conf_interval and (best_score is None or (score[1] > best_score[1] and score[2] > best_score[2])):
                n_met_constraints += 1
                best_score = score
                best_feat_combo = feat_combo
                if len(best_feat_combos) < k:
                    best_feat_combos.append(feat_combo)
                else:
                    best_feat_combos[depth] = feat_combo
                print(
                    "new best {} score: {}, from feature-set combo: {}".format(
                        condition_no_and_pvals_and_rsq_and_adjrsq_and_rmse_and_delta_rmse
                        , best_score
                        , best_feat_combo
                    )
                )
                data = [
                    {
                        'n_features': len(feat_combo)
                        , 'features': feat_combo
                        , condition_no: score[0]
                        , rsquared: score[1]
                        , adjusted_rsquared: score[2]
                        , pvals: score[3]
                        , rmse: score[4]
                        , delta_rmse: score[5]
                    }
                ]
                mask = scores_df['n_features']==k
                if len(scores_df.loc[mask]) == 0:
                    scores_df = scores_df.append(data, ignore_index=True, sort=False)
                else:
                    keys = list(data[0].keys())
                    replacement_vals = list(data[0].values())
                    scores_df.loc[mask, keys] = [replacement_vals]
        
        if n_discarded > 0:
            display(
                HTML(
                    "<p>DISCARDED {} {}-feature combinations that were not based on prior optimal feature-combo {}".format(
                        n_discarded
                        , k
                        , last_best_feat_combo
                    )
                )
            )
        if n_met_constraints > 0:
            display(
                HTML(
                    "<p>cv_selection chose the best of {} {}-feature combinations that met the constraints (out of {} considered)".format(
                        n_met_constraints
                        , k
                        , n_choose_k - n_discarded
                    )
                )
            )
        if n_choose_k - n_discarded - n_met_constraints > 0:
            display(
                HTML(
                    "<p>{} {}-feature combinations (out of {} considered) failed to meet the constraints<p><br><br>".format(
                        n_choose_k - n_discarded - n_met_constraints
                        , k
                        , n_choose_k - n_discarded
                    )
                )
            )
    
    display(HTML("<h2>Table of cv_selected Optimized Feature Combinations</h2>"))
    print_df(scores_df)

    display(HTML("<h4>cv_selected best {} = {}</h4>".format(condition_no_and_pvals_and_rsq_and_adjrsq_and_rmse_and_delta_rmse, best_score)))
    display(
        HTML(
            "<h4>cv_selected best feature-set combo ({} of {} features) {} based on {} scoring method with target cond. no. {}<h/4>".format(
                len(best_feat_combo)
                , len(base_feature_set)
                , best_feat_combo
                , condition_no_and_pvals_and_rsq_and_adjrsq_and_rmse_and_delta_rmse
                , target_cond_no
            )
        )
    )
    display(HTML("<h4>starting feature-set:{}</h4>".format(base_feature_set)))
    to_drop = list(set(base_feature_set).difference(set(best_feat_combo)))
    display(HTML("<h4>cv_selection suggests dropping {}.</h4>".format(to_drop if len(to_drop)>0 else "<i>no features</i> from {}".format(base_feature_set))))

    return (scores_df, best_feat_combo, best_score, to_drop)