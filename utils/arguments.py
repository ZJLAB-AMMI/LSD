def model_args(parser):
    group = parser.add_argument_group('Model', 'Arguments control Model')
    group.add_argument('--arch', default='LeNet', type=str, choices=['ResNet', 'LeNet'], help='model architecture')
    group.add_argument('--depth', default=5, type=int, choices=[20, 5], help='depth of the model')  # 'ResNet':20 'LeNet' 5
    group.add_argument('--model_num', default=3, type=int, help='number of submodels within the ensemble')
    group.add_argument('--model_file', default='', type=str, help='Path to the file that contains model checkpoints')
    group.add_argument('--seed', default=0, type=int, help='random seed for torch')
    group.add_argument('--gpu', default='0', type=str, help='gpu id')


def data_args(parser):
    group = parser.add_argument_group('Data', 'Arguments control Data and loading for training')
    group.add_argument('--data-dir', type=str, default='./data', help='Dataset directory')
    group.add_argument('--batch-size', type=int, default=128, help='batch size of the train loader')
    group.add_argument('--num_classes', default=10, type=int, help='num_classes')  # 10，100
    group.add_argument('--dataset', default='fmnist', type=str, choices=["cifar10", "mnist", 'fmnist'], help='dataset')


def base_train_args(parser):
    group = parser.add_argument_group('Base Training', 'Base arguments to configure training')
    group.add_argument('--epochs', default=30, type=int, choices=[200, 20, 30], help='number of training epochs')  # {"cifar10": 200, 'fmnist':30, 'mnist':20}
    group.add_argument('--lr', default=0.001, type=float, help='learning rate')
    group.add_argument('--weight_decay', default=0.0001, type=float, help='weight_decay')

    group.add_argument('--sch-intervals', nargs='*', default=[100, 150], type=int, help='learning scheduler milestones')  # "cifar10,mnist":[100, 150],fmnist:[10, 20]
    group.add_argument('--lr-gamma', default=0.1, type=float, help='learning rate decay ratio')

    group.add_argument('--adv-eps', default=8. / 255., type=float, help='perturbation budget for adversarial training')
    group.add_argument('--adv-alpha', default=2. / 255., type=float, help='step size for adversarial training')
    group.add_argument('--adv-steps', default=10, type=int, help='number of steps for adversarial training')

    group.add_argument('--plus-adv', default=False, action="store_true", help='whether perform adversarial training in the mean time with diversity training')


def adp_train_args(parser):
    group = parser.add_argument_group('ADP Training', 'Arguments to configure ADP training')
    group.add_argument('--alpha', default=2.0, type=float, help='coefficient for ensemble entropy')
    group.add_argument('--beta', default=0.5, type=float, help='coefficient for log determinant')


def dverge_train_args(parser):
    group = parser.add_argument_group('DVERGE Training', 'Arguments to configure DVERGE training')
    group.add_argument('--distill-eps', default=8. / 255., type=float, help='perturbation budget for distillation')  # 0.07
    group.add_argument('--distill-alpha', default=2. / 255., type=float, help='step size for distillation')  # 0.007
    group.add_argument('--distill-steps', default=10, type=int, help='number of steps for distillation')

    group.add_argument('--distill-fixed-layer', default=False, action="store_true", help='whether fixing the layer for distillation')
    group.add_argument('--distill-layer', default=20, type=int, help='which layer is used for distillation, only useful when distill-fixed-layer is True')  # 'ResNet':20
    group.add_argument('--distill-rand-start', default=False, action="store_true", help='whether use random start for distillation')
    group.add_argument('--distill-no-momentum', action="store_false", dest='distill_momentum', help='whether use momentum for distillation')
    group.add_argument('--dverge-coeff', default=1., type=float, help='the coefficient to balance diversity training and adversarial training')
    group.add_argument('--start-from', default='scratch', type=str, choices=['baseline', 'scratch'], help='starting point of the training')


def gal_train_args(parser):
    group = parser.add_argument_group('GAL Training', 'Arguments to configure GAL training')
    group.add_argument('--alpha', default=0.5, type=float, help='coefficient for coherence')


def lcm_gal_train_args(parser):
    group = parser.add_argument_group('lcm_gal Training', 'Arguments to configure lcm_gal training')
    group.add_argument('--ld_coeff', default=4, type=float, help='coefficient for simulated_label_distribution')
    group.add_argument('--alpha', default=2, type=float, help='coefficient for jsd_loss')
    group.add_argument('--beta', default=4, type=float, help='coefficient for coherence')  #


def trs_train_args(parser):
    group = parser.add_argument_group('trs Training', 'Arguments to configure trs training')
    group.add_argument('--scale', default=5.0, type=float)
    group.add_argument('--coeff', default=2.0, type=float, help='coefficient for cos_loss')
    group.add_argument('--lamda', default=2.0, type=float, help='coefficient for smooth_loss')


def wbox_eval_args(parser):
    group = parser.add_argument_group('White-box Evaluation', 'Arguments to configure evaluation of white-box robustness')
    group.add_argument('--subset-num', default=1000, type=int, help='number of samples of the subset, will use the full test set if none')
    group.add_argument('--random-start', default=5, type=int, help='number of random starts for PGD')
    group.add_argument('--steps', default=50, type=int, help='number of steps for PGD')
    group.add_argument('--loss-fn', default='xent', type=str, choices=['xent', 'cw'], help='which loss function to use')
    group.add_argument('--cw-conf', default=.1, type=float, help='confidence for cw loss function')
    group.add_argument('--save-to-csv', action="store_true", help='whether save the results to a csv file')
    group.add_argument('--overwrite', action="store_false", dest="append_out", help='when saving results, whether use append mode')
    group.add_argument('--convergence-check', action="store_true", help='whether perform sanity check to make sure the attack converges')
    group.add_argument("--attack_type", type=str, default='pgd', help="choose from [fgsm, pgd, cw, mim, bim, jsma, ela]")


def bbox_eval_args(parser):
    group = parser.add_argument_group('Black-box Evaluation', 'Arguments to configure evaluation of black-box robustness')
    group.add_argument('--folder', default='transfer', type=str, help='name of the folder that contains transfer adversarial examples')
    group.add_argument('--steps', default=50, type=int, help='number of PGD steps for convergence check')
    group.add_argument('--which-ensemble', default='baseline', choices=['baseline', 'dverge', 'adp', 'gal'], help='transfer from which ensemble')
    group.add_argument('--save-to-csv', action="store_true", help='whether save the results to a csv file')
    group.add_argument('--overwrite', action="store_false", dest="append_out", help='when saving results, whether use append mode')
    group.add_argument("--attack_type", type=str, default='pgd', help="choose from [fgsm, pgd, cw, mim, bim, jsma, ela]")
    group.add_argument('--subset-num', default=1000, type=int, help='number of samples of the subset')


def transf_eval_args(parser):
    group = parser.add_argument_group('Transferability Evaluation', 'Arguments to configure evaluation of transferablity among submodels')
    group.add_argument('--subset-num', default=1000, type=int, help='number of samples of the subset')
    group.add_argument('--random-start', default=5, type=int, help='number of random starts for PGD')
    group.add_argument('--steps', default=50, type=int, help='number of steps for PGD')  # 50
    group.add_argument('--save-to-csv', action="store_true", help='whether save the results to a csv file')
    group.add_argument("--attack_type", type=str, default='pgd', help="choose from [fgsm, pgd, mim, bim, jsma, cw, ela]")
