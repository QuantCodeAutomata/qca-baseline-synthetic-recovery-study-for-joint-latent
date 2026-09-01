from exp.exp1_recovery import run_exp1
from exp.exp2_product_marginals import run_exp2

if __name__ == '__main__':
    results = run_exp1(verbose=True)
    run_exp2(gamma_hat=results['gamma_hat'], p_hat=results['p_hat'], verbose=True)
    print('All experiments completed. Results saved to results/')
