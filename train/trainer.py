import math
import os
import random
import sys

sys.path.append(os.getcwd())
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(parentdir)

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from models.ensemble import Ensemble
from models.lcm import LCM
from utils import utils_models
from utils.distillation import Linf_PGD, Linf_distillation


class Trainer():
    def __init__(self, models, trainloader, testloader, writer, save_root=None, **kwargs):
        self.models = models
        self.epochs = kwargs['epochs']
        self.trainloader = trainloader
        self.testloader = testloader

        self.writer = writer
        self.save_root = save_root

        self.log_offset = 1e-6
        self.det_offset = 1e-6
        self.num_classes = kwargs['num_classes']
        self.plus_adv = kwargs['plus_adv']

        self.criterion = nn.CrossEntropyLoss()

    def get_epoch_iterator(self):
        iterator = tqdm(list(range(1, self.epochs + 1)), total=self.epochs, desc='Epoch', leave=False, position=1)
        return iterator

    def get_batch_iterator(self):
        iterator = tqdm(self.trainloader, desc='Batch', leave=False, position=2)
        return iterator

    def test(self, epoch):
        for m in self.models:
            m.eval()

        ensemble = Ensemble(self.models)

        loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for _, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.cuda(), targets.cuda()

                outputs = ensemble(inputs)
                loss += self.criterion(outputs, targets).item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()

                total += inputs.size(0)

        self.writer.add_scalar('test/ensemble_loss', loss / len(self.testloader), epoch)
        self.writer.add_scalar('test/ensemble_acc', 100 * correct / total, epoch)

        print_message = 'EnsembleLoss {loss:.4f} Acc {acc:.2%}'.format(loss=loss / len(self.testloader), acc=correct / total)
        tqdm.write(print_message)

    def save(self, epoch):
        state_dict = {}
        for i, m in enumerate(self.models):
            state_dict['model_%d' % i] = m.state_dict()
        torch.save(state_dict, os.path.join(self.save_root, 'epoch_%d.pth' % epoch))


class Baseline_Trainer(Trainer):
    def __init__(self, models, optimizers, schedulers, trainloader, testloader, writer, save_root=None, **kwargs):
        super(Baseline_Trainer, self).__init__(models, trainloader, testloader, writer, save_root, **kwargs)
        self.optimizers = optimizers
        self.schedulers = schedulers

    def run(self):
        for epoch in range(0, self.epochs + 1):
            self.train(epoch)
            self.test(epoch)
            if epoch % self.epochs == 0 and epoch != 0:
                self.save(epoch)

    def train(self, epoch):
        for m in self.models:
            m.train()

        losses = [0 for i in range(len(self.models))]

        batch_iter = tqdm(self.trainloader, bar_format='{desc} {n_fmt:>4s}/{total_fmt:<4s} {percentage:3.0f}%|{bar}| {postfix}')
        for batch_idx, (inputs, targets) in enumerate(batch_iter):
            batch_iter.set_description_str(f"Epoch {epoch + 1:04d}")
            inputs, targets = inputs.cuda(), targets.cuda()

            for i, m in enumerate(self.models):
                outputs = m(inputs)
                loss = self.criterion(outputs, targets)
                losses[i] += loss.item()

                self.optimizers[i].zero_grad()
                loss.backward()
                self.optimizers[i].step()

        print_message = 'Epoch [%3d] | ' % epoch
        for i in range(len(self.models)):
            print_message += 'Model{i:d}: {loss:.4f}  '.format(i=i + 1, loss=losses[i] / (batch_idx + 1))
        tqdm.write(print_message)

        for i in range(len(self.models)):
            self.schedulers[i].step()

        loss_dict = {}
        for i in range(len(self.models)):
            loss_dict[str(i)] = losses[i] / len(self.trainloader)
        self.writer.add_scalars('train/loss', loss_dict, epoch)


class Adversarial_Trainer(Trainer):
    def __init__(self, models, optimizers, schedulers, trainloader, testloader, writer, save_root=None, **kwargs):
        super(Adversarial_Trainer, self).__init__(models, trainloader, testloader, writer, save_root, **kwargs)
        self.optimizers = optimizers
        self.schedulers = schedulers

        # PGD configs
        self.attack_cfg = {'eps': kwargs['adv_eps'], 'alpha': kwargs['adv_alpha'], 'steps': kwargs['adv_steps'], 'is_targeted': False, 'rand_start': True}

    def run(self):
        for epoch in range(0, self.epochs + 1):
            self.train(epoch)
            self.test(epoch)
            if epoch % self.epochs == 0 and epoch != 0:
                self.save(epoch)

    def train(self, epoch):
        for m in self.models:
            m.train()

        losses = [0 for i in range(len(self.models))]

        batch_iter = tqdm(self.trainloader, bar_format='{desc} {n_fmt:>4s}/{total_fmt:<4s} {percentage:3.0f}%|{bar}| {postfix}')
        for batch_idx, (inputs, targets) in enumerate(batch_iter):
            batch_iter.set_description_str(f"Epoch {epoch + 1:04d}")
            inputs, targets = inputs.cuda(), targets.cuda()

            ensemble = Ensemble(self.models)
            adv_inputs = Linf_PGD(ensemble, inputs, targets, **self.attack_cfg)

            for i, m in enumerate(self.models):
                outputs = m(adv_inputs)
                loss = self.criterion(outputs, targets)
                losses[i] += loss.item()

                self.optimizers[i].zero_grad()
                loss.backward()
                self.optimizers[i].step()

        print_message = 'Epoch [%3d] | ' % epoch
        for i in range(len(self.models)):
            print_message += 'Model{i:d}: {loss:.4f}  '.format(i=i + 1, loss=losses[i] / (batch_idx + 1))
        tqdm.write(print_message)

        for i in range(len(self.models)):
            self.schedulers[i].step()

        loss_dict = {}
        for i in range(len(self.models)):
            loss_dict[str(i)] = losses[i] / len(self.trainloader)
        self.writer.add_scalars('train/adv_loss', loss_dict, epoch)


class ADP_Trainer(Trainer):
    def __init__(self, models, trainloader, testloader, writer, save_root=None, **kwargs):
        super(ADP_Trainer, self).__init__(models, trainloader, testloader, writer, save_root, **kwargs)
        self.log_offset = 1e-20
        self.alpha = kwargs['alpha']
        self.beta = kwargs['beta']

        params = []
        for m in self.models:
            params += list(m.parameters())
        self.optimizer = optim.Adam(params, lr=0.001, weight_decay=1e-4, eps=1e-7)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=kwargs['sch_intervals'], gamma=kwargs['lr_gamma'])

        self.criterion = nn.CrossEntropyLoss()

        self.plus_adv = kwargs['plus_adv']
        if self.plus_adv:
            self.attack_cfg = {'eps': kwargs['adv_eps'], 'alpha': kwargs['adv_alpha'], 'steps': kwargs['adv_steps'], 'is_targeted': False, 'rand_start': True}

    def run(self):
        for epoch in range(0, self.epochs + 1):
            self.train(epoch)
            self.test(epoch)
            if epoch % self.epochs == 0 and epoch != 0:
                self.save(epoch)

    def train(self, epoch):
        for m in self.models:
            m.train()

        losses = 0
        ce_losses = 0
        ee_losses = 0
        det_losses = 0
        adv_losses = 0

        batch_iter = tqdm(self.trainloader, bar_format='{desc} {n_fmt:>4s}/{total_fmt:<4s} {percentage:3.0f}%|{bar}| {postfix}')
        for batch_idx, (inputs, targets) in enumerate(batch_iter):
            batch_iter.set_description_str(f"Epoch {epoch + 1:04d}")
            inputs, targets = inputs.cuda(), targets.cuda()

            if self.plus_adv:
                ensemble = Ensemble(self.models)
                adv_inputs = Linf_PGD(ensemble, inputs, targets, **self.attack_cfg)

            # one-hot label
            y_true = torch.zeros(inputs.size(0), self.num_classes).cuda()
            y_true.scatter_(1, targets.view(-1, 1), 1)

            ce_loss = 0
            adv_loss = 0
            mask_non_y_pred = []
            ensemble_probs = 0
            for i, m in enumerate(self.models):
                outputs = m(inputs)
                ce_loss += self.criterion(outputs, targets)

                # for log_det
                y_pred = F.softmax(outputs, dim=-1)
                bool_R_y_true = torch.eq(torch.ones_like(y_true) - y_true, torch.ones_like(y_true))  # batch_size X (num_class X num_models), 2-D
                mask_non_y_pred.append(torch.masked_select(y_pred, bool_R_y_true).reshape(-1, self.num_classes - 1))  # batch_size X (num_class-1) X num_models, 1-D

                # for ensemble entropy
                ensemble_probs += y_pred

                if self.plus_adv:
                    # for adv loss
                    adv_loss += self.criterion(m(adv_inputs), targets)

            ensemble_probs = ensemble_probs / len(self.models)
            ensemble_entropy = torch.sum(-torch.mul(ensemble_probs, torch.log(ensemble_probs + self.log_offset)), dim=-1).mean()

            mask_non_y_pred = torch.stack(mask_non_y_pred, dim=1)
            assert mask_non_y_pred.shape == (inputs.size(0), len(self.models), self.num_classes - 1)
            mask_non_y_pred = mask_non_y_pred / torch.norm(mask_non_y_pred, p=2, dim=-1, keepdim=True)  # batch_size X num_model X (num_class-1), 3-D
            matrix = torch.matmul(mask_non_y_pred, mask_non_y_pred.permute(0, 2, 1))  # batch_size X num_model X num_model, 3-D
            log_det = torch.logdet(matrix + self.det_offset * torch.eye(len(self.models), device=matrix.device).unsqueeze(0)).mean()  # batch_size X 1, 1-D

            loss = ce_loss - self.alpha * ensemble_entropy - self.beta * log_det + adv_loss  # 2,0.5

            losses += loss.item()
            ce_losses += ce_loss.item()
            ee_losses += ensemble_entropy.item()
            det_losses += -log_det.item()
            if self.plus_adv:
                adv_losses += adv_loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        tqdm.write('\n ce_loss {:.4f}\t det_loss {:.4f}\t full_loss {:.4f}\n'.format(ce_loss.item(), log_det.item(), loss.item()))

        self.scheduler.step()
        self.writer.add_scalar('train/ce_loss', ce_losses / len(self.trainloader), epoch)
        self.writer.add_scalar('train/ee_loss', ee_losses / len(self.trainloader), epoch)
        self.writer.add_scalar('train/det_loss', det_losses / len(self.trainloader), epoch)
        self.writer.add_scalar('train/adv_loss', adv_losses / len(self.trainloader), epoch)
        self.writer.add_scalar('train/losses', losses / len(self.trainloader), epoch)


class DVERGE_Trainer(Trainer):
    def __init__(self, models, optimizers, schedulers, trainloader, testloader, writer, save_root=None, **kwargs):
        super(DVERGE_Trainer, self).__init__(models, trainloader, testloader, writer, save_root, **kwargs)
        self.optimizers = optimizers
        self.schedulers = schedulers

        # distillation configs
        self.distill_fixed_layer = kwargs['distill_fixed_layer']
        self.distill_cfg = {'eps': kwargs['distill_eps'], 'alpha': kwargs['distill_alpha'], 'steps': kwargs['distill_steps'], 'layer': kwargs['distill_layer'],
                            'rand_start': kwargs['distill_rand_start'], 'before_relu': True, 'momentum': kwargs['distill_momentum']}

        # diversity training configs
        self.coeff = kwargs['dverge_coeff']
        if self.plus_adv:
            self.attack_cfg = {'eps': kwargs['adv_eps'], 'alpha': kwargs['adv_alpha'], 'steps': kwargs['adv_steps'], 'is_targeted': False, 'rand_start': True}

        self.depth = kwargs['depth']

    def get_batch_iterator(self):
        loader = utils_models.DistillationLoader(self.trainloader, self.trainloader)
        iterator = tqdm(loader, desc='Batch', leave=False, position=2)
        return iterator

    def run(self):
        for epoch in range(0, self.epochs + 1):
            self.train(epoch)
            self.test(epoch)
            if epoch % self.epochs == 0 and epoch != 0:
                self.save(epoch)

    def train(self, epoch):
        for m in self.models:
            m.train()

        if not self.distill_fixed_layer:
            tqdm.write('Randomly choosing a layer for distillation...')
            self.distill_cfg['layer'] = random.randint(1, self.depth)

        losses = [0 for i in range(len(self.models))]

        loader = utils_models.DistillationLoader(self.trainloader, self.trainloader)
        batch_iter = tqdm(loader, bar_format='{desc} {n_fmt:>4s}/{total_fmt:<4s} {percentage:3.0f}%|{bar}| {postfix}')
        for batch_idx, (si, sl, ti, tl) in enumerate(batch_iter):
            batch_iter.set_description_str(f"Epoch {epoch + 1:04d}")
            si, sl = si.cuda(), sl.cuda()
            ti, tl = ti.cuda(), tl.cuda()

            if self.plus_adv:
                adv_inputs_list = []

            distilled_data_list = []
            for m in self.models:
                temp = Linf_distillation(m, si, ti, **self.distill_cfg)
                distilled_data_list.append(temp)

                if self.plus_adv:
                    temp = Linf_PGD(m, si, sl, **self.attack_cfg)
                    adv_inputs_list.append(temp)

            for i, m in enumerate(self.models):
                loss = 0

                for j, distilled_data in enumerate(distilled_data_list):
                    if i == j:
                        continue

                    outputs = m(distilled_data)
                    loss += self.criterion(outputs, sl)

                if self.plus_adv:
                    outputs = m(adv_inputs_list[i])
                    loss = self.coeff * loss + self.criterion(outputs, sl)

                losses[i] += loss.item()
                self.optimizers[i].zero_grad()
                loss.backward()
                self.optimizers[i].step()

        for i in range(len(self.models)):
            self.schedulers[i].step()

        print_message = 'Epoch [%3d] | ' % epoch
        for i in range(len(self.models)):
            print_message += 'Model{i:d}: {loss:.4f}  '.format(i=i + 1, loss=losses[i] / (batch_idx + 1))
        tqdm.write(print_message)

        loss_dict = {}
        for i in range(len(self.models)):
            loss_dict[str(i)] = losses[i] / len(self.trainloader)
        self.writer.add_scalars('train/loss', loss_dict, epoch)


class GAL_Trainer(Trainer):
    def __init__(self, models, trainloader, testloader, writer, save_root=None, **kwargs):
        super(GAL_Trainer, self).__init__(models, trainloader, testloader, writer, save_root, **kwargs)
        self.coeff = kwargs['alpha']

        params = []
        for m in self.models:
            params += list(m.parameters())
        self.optimizer = optim.Adam(params, lr=kwargs['lr'], weight_decay=kwargs['weight_decay'], eps=1e-7)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=kwargs['sch_intervals'], gamma=kwargs['lr_gamma'])

        if self.plus_adv:
            self.attack_cfg = {'eps': kwargs['adv_eps'], 'alpha': kwargs['adv_alpha'], 'steps': kwargs['adv_steps'], 'is_targeted': False, 'rand_start': True}

    def run(self):
        for epoch in range(0, self.epochs + 1):
            self.train(epoch)
            self.test(epoch)
            if epoch % self.epochs == 0 and epoch != 0:
                self.save(epoch)

    def train(self, epoch):
        for m in self.models:
            m.train()

        losses = 0
        ce_losses = 0
        coh_losses = 0
        adv_losses = 0

        batch_iter = tqdm(self.trainloader, bar_format='{desc} {n_fmt:>4s}/{total_fmt:<4s} {percentage:3.0f}%|{bar}| {postfix}')
        for batch_idx, (inputs, targets) in enumerate(batch_iter):
            batch_iter.set_description_str(f"Epoch {epoch + 1:04d}")
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs.requires_grad = True

            if self.plus_adv:
                ensemble = Ensemble(self.models)
                adv_inputs = Linf_PGD(ensemble, inputs, targets, **self.attack_cfg)

            ce_loss = 0
            adv_loss = 0
            grads = []
            for i, m in enumerate(self.models):
                # for coherence loss
                outputs = m(inputs)

                loss = self.criterion(outputs, targets)
                grad = autograd.grad(loss, inputs, create_graph=True)[0]
                grad = grad.flatten(start_dim=1)
                grads.append(grad)

                # for standard loss
                ce_loss += self.criterion(m(inputs.clone().detach()), targets)

                if self.plus_adv:
                    # for adv loss
                    adv_loss += self.criterion(m(adv_inputs), targets)

            cos_sim = []
            for i in range(len(self.models)):
                for j in range(i + 1, len(self.models)):
                    cos_sim.append(F.cosine_similarity(grads[i], grads[j], dim=-1))

            cos_sim = torch.stack(cos_sim, dim=-1)
            assert cos_sim.shape == (inputs.size(0), (len(self.models) * (len(self.models) - 1)) // 2)
            coh_loss = torch.log(cos_sim.exp().sum(dim=-1) + self.log_offset).mean()

            loss = ce_loss / len(self.models) + self.coeff * coh_loss + adv_loss / len(self.models)

            losses += loss.item()
            ce_losses += ce_loss.item()
            coh_losses += coh_loss.item()
            if self.plus_adv:
                adv_losses += adv_loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        tqdm.write('\n ce_loss {:.4f}\t coh_loss {:.4f}\t full_loss {:.4f}\n'.format(ce_loss.item() / len(self.models), self.coeff * coh_loss.item(), loss.item()))

        self.scheduler.step()
        self.writer.add_scalar('train/ce_loss', ce_losses / len(self.trainloader), epoch)
        self.writer.add_scalar('train/coh_loss', coh_losses / len(self.trainloader), epoch)
        self.writer.add_scalar('train/adv_loss', adv_losses / len(self.trainloader), epoch)
        self.writer.add_scalar('train/losses', losses / len(self.trainloader), epoch)


class jsd_loss(nn.Module):
    def __init__(self, num_models):
        super(jsd_loss, self).__init__()
        self.num_models = num_models
        self.criterion_kl = nn.KLDivLoss()
        self.log_offset = 1e-6

    def forward(self, mask_label):
        jsd_sim = []
        for i in range(self.num_models):
            for j in range(i + 1, self.num_models):
                P = mask_label[i] + 1e-30  # offset
                Q = mask_label[j] + 1e-30
                M = (P + Q) / 2

                kl_left = self.criterion_kl(torch.log(M), P) / math.log(2)  # torch.log(M) uses the base e logarithm
                kl_right = self.criterion_kl(torch.log(M), Q) / math.log(2)
                jsd_value = (kl_left + kl_right) / 2

                jsd_sim.append(jsd_value)

        jsd_sim = torch.stack(jsd_sim, dim=-1)
        jsd_loss = torch.log(jsd_sim.exp().sum(dim=-1) / len(jsd_sim) + self.log_offset).mean()

        return jsd_loss


class LCM_Trainer(Trainer):
    def __init__(self, models, trainloader, testloader, writer, save_root=None, **kwargs):
        super(LCM_Trainer, self).__init__(models, trainloader, testloader, writer, save_root, **kwargs)
        self.alpha = kwargs['alpha']
        self.coeff = kwargs['ld_coeff']
        self.lcm_model = LCM(n_input=self.num_classes, n_hidden=64, n_output=self.num_classes).cuda()
        self.criterion_jsd = jsd_loss(num_models=len(self.models)).cuda()
        self.criterion_kl = nn.KLDivLoss()

        params = []
        for m in self.models:
            params += list(m.parameters())
        params += list(self.lcm_model.parameters())

        self.optimizer = optim.Adam(params, lr=kwargs['lr'], weight_decay=kwargs['weight_decay'], eps=1e-7)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=kwargs['sch_intervals'], gamma=kwargs['lr_gamma'])

        if self.plus_adv:
            self.attack_cfg = {'eps': kwargs['adv_eps'], 'alpha': kwargs['adv_alpha'], 'steps': kwargs['adv_steps'], 'is_targeted': False, 'rand_start': True}

    def run(self):
        for epoch in range(0, self.epochs + 1):
            self.train(epoch)
            self.test(epoch)
            if epoch % self.epochs == 0 and epoch != 0:
                self.save(epoch)

                state_dict = {}
                state_dict['lcm_model'] = self.lcm_model.state_dict()
                torch.save(state_dict, os.path.join(self.save_root, 'lcm_model_epoch_%d.pth' % epoch))

    def train(self, epoch):
        for m in self.models:
            m.train()
        self.lcm_model.train()

        losses = 0
        lcm_losses = 0
        jsd_losses = 0
        adv_losses = 0

        batch_iter = tqdm(self.trainloader, bar_format='{desc} {n_fmt:>4s}/{total_fmt:<4s} {percentage:3.0f}%|{bar}| {postfix}')
        for batch_idx, (inputs, targets) in enumerate(batch_iter):
            batch_iter.set_description_str(f"Epoch {epoch + 1:04d}")
            inputs, targets = inputs.cuda(), targets.cuda()
            L_train = torch.tensor(np.array([np.array(range(self.num_classes)) for i in range(len(inputs))]), dtype=torch.int64).cuda()

            if self.plus_adv:
                ensemble = Ensemble(self.models)
                adv_inputs = Linf_PGD(ensemble, inputs, targets, **self.attack_cfg)

            # one-hot label
            y_true = torch.zeros(inputs.size(0), self.num_classes).cuda()
            y_true.scatter_(1, targets.view(-1, 1), 1)  # 128*10

            lcm_loss = 0
            adv_loss = 0
            mask_label = []
            for i, m in enumerate(self.models):
                pred_probs = m(inputs)  # 128*10
                pred_probs = F.softmax(pred_probs, dim=-1)
                input_vec = m.get_input_vec(inputs)  # 128*64

                label_confusion_vector = self.lcm_model(labels=L_train, input_vec=input_vec)  # 128*10
                simulated_label_distribution = F.softmax((label_confusion_vector + self.coeff * y_true), dim=-1).type(torch.float64).cuda()  # 128*10
                sld_loss = torch.sum(-torch.mul(simulated_label_distribution, torch.log(pred_probs + self.log_offset)), dim=-1).mean()
                sld_entropy = torch.sum(-torch.mul(simulated_label_distribution, torch.log(simulated_label_distribution + self.log_offset)), dim=-1).mean()

                lcm_loss += (sld_loss - sld_entropy)

                # for jsd_loss
                bool_R_y_true = torch.eq(torch.ones_like(y_true) - y_true, torch.ones_like(y_true))
                mask_label.append(torch.masked_select(label_confusion_vector, bool_R_y_true).reshape(-1, self.num_classes - 1))

                if self.plus_adv:
                    # for adv loss
                    # adv_loss += self.criterion(m(adv_inputs), targets)

                    adv_outputs = m(adv_inputs)  # 128*10
                    adv_input_vec = m.get_input_vec(adv_inputs)  # 128*64
                    adv_label_confusion_vector = self.lcm_model(labels=L_train, input_vec=adv_input_vec)  # 128*10
                    adv_simulated_label_distribution = F.softmax((adv_label_confusion_vector + self.coeff * y_true), dim=-1).type(torch.float64).cuda()  # 128*10
                    adv_pred_probs = F.softmax(adv_outputs, dim=-1)
                    adv_sld_loss = torch.sum(-torch.mul(adv_simulated_label_distribution, torch.log(adv_pred_probs + self.log_offset)), dim=-1).mean()
                    adv_sld_entropy = torch.sum(-torch.mul(adv_simulated_label_distribution, torch.log(adv_simulated_label_distribution + self.log_offset)), dim=-1).mean()
                    adv_loss += (adv_sld_loss - adv_sld_entropy)

            jsd_loss = self.criterion_jsd(mask_label)

            loss = lcm_loss / len(self.models) - self.alpha * jsd_loss + adv_loss / len(self.models)

            losses += loss.item()
            lcm_losses += lcm_loss.item()
            jsd_losses += jsd_loss.item()
            if self.plus_adv:
                adv_losses += adv_loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        tqdm.write('\nlcm_loss {:.4f}\t jsd_loss {:.4f}\t full_loss {:.4f}'.format(lcm_loss.item() / len(self.models), self.alpha * jsd_loss.item(), loss.item()))

        self.scheduler.step()
        self.writer.add_scalar('train/lcm_losses', lcm_losses / len(self.trainloader), epoch)
        self.writer.add_scalar('train/jsd_loss', jsd_losses / len(self.trainloader), epoch)
        self.writer.add_scalar('train/adv_loss', adv_losses / len(self.trainloader), epoch)
        self.writer.add_scalar('train/losses', losses / len(self.trainloader), epoch)


class LCM_GAL_Trainer(LCM_Trainer):
    '''
    the implementation of our Ensemble Diversity Training
    '''

    def __init__(self, models, trainloader, testloader, writer, save_root=None, **kwargs):
        super(LCM_GAL_Trainer, self).__init__(models, trainloader, testloader, writer, save_root, **kwargs)
        self.beta_gal = kwargs['beta']
        self.alpha_lcm = kwargs['alpha']
        self.coeff_lcm = kwargs['ld_coeff']

    def run(self):
        for epoch in range(0, self.epochs + 1):
            self.train(epoch)
            self.test(epoch)
            if epoch % self.epochs == 0 and epoch != 0:
                self.save(epoch)

                state_dict = {}
                state_dict['lcm_model'] = self.lcm_model.state_dict()
                torch.save(state_dict, os.path.join(self.save_root, 'lcm_model_epoch_%d.pth' % epoch))

    def train(self, epoch):
        for m in self.models:
            m.train()
        self.lcm_model.train()

        losses = 0
        lcm_losses = 0
        jsd_losses = 0
        coh_losses = 0
        adv_losses = 0
        batch_iter = tqdm(self.trainloader, bar_format='{desc} {n_fmt:>4s}/{total_fmt:<4s} {percentage:3.0f}%|{bar}| {postfix}')
        for batch_idx, (inputs, targets) in enumerate(batch_iter):
            batch_iter.set_description_str(f"Epoch {epoch + 1:04d}")
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs.requires_grad = True

            L_train = torch.tensor(np.array([np.array(range(self.num_classes)) for i in range(len(inputs))]), dtype=torch.int64).cuda()
            if self.plus_adv:
                ensemble = Ensemble(self.models)
                adv_inputs = Linf_PGD(ensemble, inputs, targets, **self.attack_cfg)

            # one-hot label
            y_true = torch.zeros(inputs.size(0), self.num_classes).cuda()
            y_true.scatter_(1, targets.view(-1, 1), 1)  # 128*10

            lcm_loss = 0
            adv_loss = 0
            mask_label = []
            grads = []
            for i, m in enumerate(self.models):
                outputs = m(inputs)  # 128*10

                input_vec = m.get_input_vec(inputs)  # 128*64
                label_confusion_vector = self.lcm_model(labels=L_train, input_vec=input_vec)  # 128*10
                simulated_label_distribution = F.softmax((label_confusion_vector + self.coeff_lcm * y_true), dim=-1).type(torch.float64).cuda()  # 128*10

                pred_probs = F.softmax(outputs, dim=-1)

                # for standard loss
                sld_loss = torch.sum(-torch.mul(simulated_label_distribution, torch.log(pred_probs + self.log_offset)), dim=-1).mean()
                sld_entropy = torch.sum(-torch.mul(simulated_label_distribution, torch.log(simulated_label_distribution + self.log_offset)), dim=-1).mean()
                loss_t = (sld_loss - sld_entropy)
                lcm_loss += loss_t

                # for gal_loss
                # loss_t = self.criterion(m(inputs.clone().detach()), targets)
                grad = autograd.grad(loss_t, inputs, create_graph=True)[0]
                grad = grad.flatten(start_dim=1)
                grads.append(grad)

                # for jsd_loss
                bool_R_y_true = torch.eq(torch.ones_like(y_true) - y_true, torch.ones_like(y_true))
                mask_label.append(torch.masked_select(label_confusion_vector, bool_R_y_true).reshape(-1, self.num_classes - 1))

                if self.plus_adv:
                    # for adv loss
                    # adv_loss += self.criterion(m(adv_inputs), targets)

                    adv_outputs = m(adv_inputs)  # 128*10
                    adv_input_vec = m.get_input_vec(adv_inputs)  # 128*64
                    adv_label_confusion_vector = self.lcm_model(labels=L_train, input_vec=adv_input_vec)  # 128*10
                    adv_simulated_label_distribution = F.softmax((adv_label_confusion_vector + self.coeff_lcm * y_true), dim=-1).type(torch.float64).cuda()  # 128*10
                    adv_pred_probs = F.softmax(adv_outputs, dim=-1)
                    adv_sld_loss = torch.sum(-torch.mul(adv_simulated_label_distribution, torch.log(adv_pred_probs + self.log_offset)), dim=-1).mean()
                    adv_sld_entropy = torch.sum(-torch.mul(adv_simulated_label_distribution, torch.log(adv_simulated_label_distribution + self.log_offset)), dim=-1).mean()
                    adv_loss += (adv_sld_loss - adv_sld_entropy)

            jsd_loss = self.criterion_jsd(mask_label)

            # coh_loss_V1
            # cos_sim = []
            # for i in range(len(self.models)):
            #     for j in range(i + 1, len(self.models)):
            #         cos_sim.append(F.cosine_similarity(grads[i], grads[j], dim=-1))
            # cos_sim = torch.stack(cos_sim, dim=-1)
            # coh_loss = torch.log(cos_sim.exp().sum(dim=-1) + self.log_offset).mean() / len(self.models)

            # coh_loss_V2
            cos_sim = []
            for i in range(len(self.models)):
                for j in range(i + 1, len(self.models)):
                    cos_sim.append(torch.abs(F.cosine_similarity(grads[i], grads[j], dim=-1)).mean())
            cos_sim = torch.stack(cos_sim, dim=-1)
            coh_loss = cos_sim.sum(dim=-1).mean() / len(self.models)

            loss = lcm_loss / len(self.models) + adv_loss / len(self.models) - self.alpha_lcm * jsd_loss + self.beta_gal * coh_loss

            losses += loss.item()
            lcm_losses += lcm_loss.item()
            jsd_losses += jsd_loss.item()
            coh_losses += coh_loss.item()
            if self.plus_adv:
                adv_losses += adv_loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        tqdm.write(
            '\nlcm_loss {:.4f}\t jsd_loss {:.4f}\t coh_loss {:.4f}\t full_loss {:.4f}'.format(lcm_loss.item() / len(self.models), self.alpha_lcm * jsd_loss.item(), self.beta_gal * coh_loss.item(),
                                                                                              loss.item()))

        self.scheduler.step()
        self.writer.add_scalar('train/losses', losses / len(self.trainloader), epoch)
        self.writer.add_scalar('train/lcm_losses', lcm_losses / len(self.trainloader), epoch)
        self.writer.add_scalar('train/jsd_loss', jsd_losses / len(self.trainloader), epoch)
        self.writer.add_scalar('train/coh_loss', coh_losses / len(self.trainloader), epoch)
        self.writer.add_scalar('train/adv_loss', adv_losses / len(self.trainloader), epoch)


class TRS_Trainer(Trainer):
    def __init__(self, models, trainloader, testloader, writer, save_root=None, **kwargs):
        super(TRS_Trainer, self).__init__(models, trainloader, testloader, writer, save_root, **kwargs)
        self.scale = kwargs['scale']
        self.coeff = kwargs['coeff']
        self.lamda = kwargs['lamda']
        params = []
        for m in self.models:
            params += list(m.parameters())

        self.optimizer = optim.Adam(params, lr=kwargs['lr'], weight_decay=kwargs['weight_decay'], eps=1e-7)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=kwargs['sch_intervals'], gamma=kwargs['lr_gamma'])

        if self.plus_adv:
            self.attack_cfg = {'eps': kwargs['adv_eps'], 'alpha': kwargs['adv_alpha'], 'steps': kwargs['adv_steps'], 'is_targeted': False, 'rand_start': True}

    def run(self):
        for epoch in range(0, self.epochs + 1):
            self.train(epoch)
            self.test(epoch)
            if epoch % self.epochs == 0 and epoch != 0:
                self.save(epoch)

    def train(self, epoch):
        for m in self.models:
            m.train()

        losses = 0
        cos_losses = 0
        adv_losses = 0
        smooth_losses = 0

        batch_iter = tqdm(self.trainloader, bar_format='{desc} {n_fmt:>4s}/{total_fmt:<4s} {percentage:3.0f}%|{bar}| {postfix}')
        for batch_idx, (inputs, targets) in enumerate(batch_iter):
            batch_iter.set_description_str(f"Epoch {epoch + 1:04d}")
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs.requires_grad = True

            if self.plus_adv:
                ensemble = Ensemble(self.models)
                adv_inputs = Linf_PGD(ensemble, inputs, targets, **self.attack_cfg)
                adv_inputs.requires_grad = True

            loss_std = 0
            adv_loss = 0
            smooth_loss = 0
            grads = []
            for i, m in enumerate(self.models):
                logits = m(inputs)

                # for cos_loss
                loss = self.criterion(logits, targets)
                grad = autograd.grad(loss, inputs, create_graph=True)[0]
                grad = grad.flatten(start_dim=1)
                grads.append(grad)

                # for standard loss
                loss_std += loss

                # for smooth_loss
                smooth_loss += ((torch.sum(grad ** 2, 1)).mean() * 2)

                if self.plus_adv:
                    # for adv loss
                    logits = m(adv_inputs)
                    loss = self.criterion(logits, targets)
                    adv_loss += loss

                    grad = autograd.grad(loss, adv_inputs, create_graph=True)[0]
                    grad = grad.flatten(start_dim=1)
                    smooth_loss += ((torch.sum(grad ** 2, 1)).mean() * 2)

            # cos_loss
            cos_sim = []
            for i in range(len(self.models)):
                for j in range(i + 1, len(self.models)):
                    cos_sim.append(torch.abs(F.cosine_similarity(grads[i], grads[j], dim=-1)).mean())
            cos_sim = torch.stack(cos_sim, dim=-1)
            cos_loss = cos_sim.sum(dim=-1).mean() / len(cos_sim)

            smooth_loss /= len(self.models)
            loss = loss_std / len(self.models) + self.scale * (self.coeff * cos_loss + self.lamda * smooth_loss)

            losses += loss.item()
            cos_losses += cos_loss.item()
            smooth_losses += smooth_loss.item()
            if self.plus_adv:
                adv_losses += adv_loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        tqdm.write(
            '\nloss_std {:.4f}\t cos_loss {:.4f}\t smooth_loss {:.4f}\t full_loss {:.4f}'.format(loss_std.item() / len(self.models), self.coeff * cos_loss.item(), self.lamda * smooth_loss.item(),
                                                                                                 loss.item()))

        self.scheduler.step()
        self.writer.add_scalar('train/losses', losses / len(self.trainloader), epoch)
        self.writer.add_scalar('train/cos_loss', cos_losses / len(self.trainloader), epoch)
        self.writer.add_scalar('train/smooth_loss', smooth_losses / len(self.trainloader), epoch)
        self.writer.add_scalar('train/adv_loss', adv_losses / len(self.trainloader), epoch)
