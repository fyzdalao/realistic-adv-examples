import argparse

import numpy as np
import torch

from src.attack_results import AttackResults
from src.setup import setup_attack, setup_model_and_data, setup_out_dir


def main(args):
    targeted = True if args.targeted == '1' else False
    early_stopping = False if args.early == '0' else True

    print(args)

    device = torch.device("cuda")

    model, test_loader = setup_model_and_data(args, device)
    exp_out_dir = setup_out_dir(args)
    attack = setup_attack(args)
    attack_results = AttackResults(exp_out_dir)

    seeds = np.random.randint(10000, size=10000)

    count = 0
    misclassified = 0
    negatives = 0

    for i, batch in enumerate(test_loader):
        if count == args.num:
            break

        if isinstance(batch, dict):
            xi, yi = batch["image"], batch["label"]
        else:
            xi, yi = batch

        print(f"Sample {i}, class: {yi.item()}, attacks count: {count}")
        xi, yi = xi.to(device), yi.to(device)

        if model.n_class == 2 and yi.item() == 0:
            negatives += 1
            print("Skipping as item is negative")
            continue

        if model.predict_label(xi) != yi:
            misclassified += 1
            print("Skipping as item is misclassified")
            continue

        np.random.seed(seeds[count])

        target = np.random.randint(model.n_class) * torch.ones(yi.shape,
                                                               dtype=torch.long).to(device) if targeted else None
        while target and torch.sum(target == yi) > 0:
            print('re-generate target label')
            target = np.random.randint(model.n_class) * torch.ones(len(xi), dtype=torch.long).to(device)

        model.num_queries = 0
        with torch.no_grad():
            adv, queries_counter, dist, succ, extra_results = attack(model, xi, yi, target)

        if args.save_img_every is not None and count % args.save_img_every == 0:
            np.save(exp_out_dir / f"{i}_adv.npy", adv[0].cpu().numpy())
            np.save(exp_out_dir / f"{i}.npy", xi[0].cpu().numpy())

        if succ or not early_stopping:
            attack_results = attack_results.update_with_success(dist, queries_counter, extra_results)
        else:
            attack_results = attack_results.update_with_failure(dist, queries_counter, extra_results)

        attack_results.log_results(count)
        attack_results.save_results(verbose=True)
        count += 1
        # if attack_results.has_simulated_counters:
        #     print("Simulated results:")
        #     attack_results.simulated_self.log_results(i)

    attack_results.save_results(verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hard Label Attacks')
    parser.add_argument('--attack', default='rays', type=str, help='The attack to run')
    parser.add_argument('--dataset', default='imagenet', type=str, help='Dataset')
    parser.add_argument('--targeted', default='0', type=str, help='targeted or untargeted')
    parser.add_argument('--norm', default='linf', type=str, help='Norm for attack, linf only')
    parser.add_argument('--num', default=1000, type=int, help='Number of samples to be attacked from test dataset.')
    parser.add_argument('--max-queries', default=None, type=int, help='Maximum queries for the attack')
    parser.add_argument('--max-unsafe-queries', default=None, type=int, help='Maximum unsafe queries for the attack')
    parser.add_argument('--batch', default=1, type=int, help='attack batch size.')
    parser.add_argument('--epsilon', default=None, type=float, help='attack strength')
    parser.add_argument('--early',
                        default='1',
                        type=str,
                        help='early stopping (stop attack once the adversarial example is found)')
    parser.add_argument('--search', default='binary', type=str, help='Type of search to use, binary or line')
    parser.add_argument('--line-search-tol',
                        default=None,
                        type=float,
                        help='Tolerance for line search w.r.t. previous iteration')
    parser.add_argument(
        '--out-dir',
        default='/local/home/edebenedetti/exp-results/realistic-adv-examples/',
        type=str,
    )
    parser.add_argument(
        '--data-dir',
        default='data/imagenet/val',
        type=str,
    )
    parser.add_argument(
        '--save-img-every',
        default=50,
        type=int,
    )
    parser.add_argument('--strong-preprocessing',
                        default='0',
                        type=str,
                        help='Whether strong preprocessing (i.e., JPEG, Resize, Crop) '
                        'should be applied before feeding the image to the classifier')
    parser.add_argument('--model-threshold', default=0.25, type=float, help='The threshold to use for the API model')
    parser.add_argument('--discrete',
                        default='0',
                        type=str,
                        help='Whether the attack should work in discrete space (i.e., int8)')
    parser.add_argument('--rays-flip-squares',
                        default='0',
                        type=str,
                        help='Whether the attack should flip squares and not chunks of a 1-d vector')
    parser.add_argument('--rays-flip-rand-pixels',
                        default='0',
                        type=str,
                        help='Whether the attack should flip random pixels not chunks of a 1-d vector')
    parser.add_argument('--max-iter', default=None, type=int, help='Number of iterations for HSJA, OPT and SignOPT')
    parser.add_argument('--hsja-stepsize-search',
                        default='geometric_progression',
                        type=str,
                        help='Stepsize search for HSJA')
    parser.add_argument('--hsja-max-num-evals', default=1e4, type=int, help='Max number of evaluations for HSJA')
    parser.add_argument('--hsja-init-num-evals', default=100, type=int, help='Max number of evaluations for HSJA')
    parser.add_argument('--hsja-gamma',
                        default=10_000,
                        type=float,
                        help='gamma parameter for HSJA (used for the binary search threshold)')
    parser.add_argument('--hsja-delta',
                        default=None,
                        type=float,
                        help='Whether to use a fixed delta for gradient estimation in HSJA, '
                        'an adaptive delta is used if this is None')
    parser.add_argument('--hsja-grad-est-mode',
                        default='hsja',
                        type=str,
                        help='Gradient estimation mode for HSJA, one of `hsja`, `opt` or `sign_opt`')
    parser.add_argument('--hsja-n-searches',
                        default=2,
                        type=int,
                        help='Number of searches for OPT and Sign OPT',
                        choices=[1, 2])
    parser.add_argument('--hsja-bias-coef',
                        default=0.0,
                        type=float,
                        help='bias coefficient for HSJA (used to move boundary point farther before gradient estimation.)')
    parser.add_argument('--hsja-lower-bad-query-bound',
                        default=10,
                        type=int,
                        help='desired lower bound on the number of bad queries in the gradient esitmation phase.')
    parser.add_argument('--hsja-upper-bad-query-bound',
                        default=20,
                        type=int,
                        help='desired upper bound on the number of bad queries in the gradient esitmation phase.')
    parser.add_argument('--hsja-bias-coef-change-rate',
                        default=0.1,
                        type=float,
                        help='used to change the bias coefficient adaptively.')
    
    parser.add_argument('--opt-alpha', default=0.2, type=float, help='alpha parameter for OPT and Sign OPT')
    parser.add_argument('--opt-beta', default=0.01, type=float, help='beta parameter for OPT and Sign OPT')
    parser.add_argument(
        '--opt-grad-est-search',
        default=None,
        type=str,
        help='What search should be used for gradient estimation in OPT. Default is what it specified with --search')
    parser.add_argument(
        '--opt-step-size-search',
        default=None,
        type=str,
        help='What search should be used for step size search in OPT. Default is what it specified with --search')
    parser.add_argument('--opt-line-search-overshoot',
                        default=2.5,
                        type=float,
                        help='Line search overshoot for OPT and Sign OPT')
    parser.add_argument('--opt-n-searches',
                        default=2,
                        type=int,
                        help='Number of searches for OPT and Sign OPT',
                        choices=[1, 2])
    parser.add_argument('--opt-max-search-steps',
                        default=10_000,
                        type=int,
                        help='Max number of line search steps for OPT and Sign OPT')
    parser.add_argument('--opt-bs', default=100, type=int, help='Batch size for for OPT and Sign OPT')
    parser.add_argument('--opt-num-grad-queries', default=10, type=int, help='Number of gradient queries for OPT')
    parser.add_argument('--opt-num-init-directions', default=100, type=int, help='Number of init directions OPT')
    parser.add_argument('--opt-get-one-init-direction',
                        default='0',
                        type=str,
                        help='Whether OPT should stop when it finds one good initial direction')
    parser.add_argument('--sign-opt-num-grad-queries',
                        default=200,
                        type=int,
                        help='Number of gradient queries for Sign OPT')
    parser.add_argument('--sign-opt-momentum', default=0., type=float, help='Momentum for Sign OPT')
    
    
    parser.add_argument('--geoda-n-searches',
                        default=2,
                        type=int,
                        help='Number of searches for GeoDA',
                        choices=[1, 2])
    parser.add_argument('--geoda-theta',
                        default=1e-4,
                        type=float,
                        help='Theta parameter for GeoDA')
    parser.add_argument('--geoda-delta',
                        default=2e-4,
                        type=float,
                        help='Delta parameter for GeoDA')
    parser.add_argument('--geoda-max-num-evals', default=1e4, type=int, help='Max number of evaluations for GeoDA')
    parser.add_argument('--geoda-init-num-evals', default=100, type=int, help='GeoDA init num evals')
    parser.add_argument('--geoda-bias-coef',
                        default=0.0,
                        type=float,
                        help='bias coefficient for GeoDA (used to move boundary point further before gradient estimation.)')
    parser.add_argument('--geoda-lower-bad-query-bound',
                        default=10,
                        type=int,
                        help='desired lower bound on the number of bad queries in the gradient esitmation phase.')
    parser.add_argument('--geoda-upper-bad-query-bound',
                        default=20,
                        type=int,
                        help='desired upper bound on the number of bad queries in the gradient esitmation phase.')
    parser.add_argument('--geoda-bias-coef-change-rate',
                        default=0.1,
                        type=float,
                        help='used to change the bias coefficient adaptively.')
    parser.add_argument('--geoda-dim-reduc-factor',
                        default=1e0,
                        type=float,
                        help='Dimension reduciton factor for creating subspace in GeoDA')
    parser.add_argument('--geoda-search-radius-increase',
                        default=1.1,
                        type=float,
                        help='Multiplier used to increase search radius')
    
    _args = parser.parse_args()
    main(_args)
