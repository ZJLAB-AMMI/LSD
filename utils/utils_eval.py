import os

from advertorch.attacks import GradientSignAttack
from advertorch.attacks import LinfBasicIterativeAttack, LinfMomentumIterativeAttack
from advertorch.attacks import LinfPGDAttack


def get_adversary_obj(args, ensemble, loss_fn, eps, attack_type):
    if (attack_type == "pgd"):
        adversary = LinfPGDAttack(ensemble, loss_fn=loss_fn, eps=eps, nb_iter=args.adv_steps, eps_iter=eps / 5, rand_init=True, clip_min=0., clip_max=1., targeted=False)
    elif (attack_type == "fgsm"):
        adversary = GradientSignAttack(ensemble, loss_fn=loss_fn, eps=eps, clip_min=0., clip_max=1., targeted=False)
    elif (attack_type == "mim"):
        adversary = LinfMomentumIterativeAttack(ensemble, loss_fn=loss_fn, eps=eps, nb_iter=args.adv_steps, eps_iter=eps / 5, clip_min=0., clip_max=1., targeted=False)
    elif (attack_type == "bim"):
        adversary = LinfBasicIterativeAttack(ensemble, loss_fn=loss_fn, eps=eps, nb_iter=args.adv_steps, eps_iter=eps / 5, clip_min=0., clip_max=1., targeted=False)
    else:
        raise ValueError('[{:s}] attack_type is not supported yet...')
    return adversary


def get_root_path(args, alg_name):
    sub_dir = '{:d}_{:s}{:d}_{:s}'.format(args.model_num, args.arch, args.depth, args.dataset)
    if alg_name == 'dverge':
        save_root = os.path.join('checkpoints', 'dverge', 'seed_{:d}'.format(args.seed), sub_dir)
        if args.start_from == 'scratch':
            save_root += '_start_from_scratch'
    elif alg_name == 'advt':
        save_root = os.path.join('checkpoints', 'advt', 'seed_{:d}'.format(args.seed), sub_dir)
    else:
        save_root = os.path.join('checkpoints', alg_name, 'seed_{:d}'.format(args.seed), sub_dir)

    return save_root
