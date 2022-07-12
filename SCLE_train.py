import math
import time
import numpy as np
import torch
import torch.nn.modules.loss
import torch.nn.functional as F
from progress.bar import Bar
from SCLE_model import SCLE, Classification
import st_loss
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def adjust_learning_rate(optimizer, epoch, lr):
    p = {
        'epochs': 500,
        'optimizer': 'sgd',
        'optimizer_kwargs': {'nesterov': False,
                             'weight_decay': 0.0001,
                             'momentum': 0.9,

                             },
        'scheduler': 'cosine',
        'scheduler_kwargs': {'lr_decay_rate': 0.1},
    }

    if p['scheduler'] == 'cosine':
        eta_min = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** 3)
        new_lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / p['epochs'])) / 2

    elif p['scheduler'] == 'step':
        steps = np.sum(epoch > np.array(p['scheduler_kwargs']['lr_decay_epochs']))
        if steps > 0:
            new_lr = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** steps)

    elif p['scheduler'] == 'constant':
        new_lr = lr

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(p['scheduler']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

    return lr


class SCLE_Train:
    def __init__(self, adata_X, params, A=None, X_sim=None, X_label=None):
        self.params = params
        self.device = params.device
        self.epochs = params.epochs
        self.X = adata_X.X          # spatial transcriptomics data
        # self.eval_loader = self.get_loader(X, args_transformation)
        self.A = A
        self.labels = np.ones((params.cell_num, 1))
        self.dims = np.concatenate([[self.X.shape[1]], params.layers])
        self.batch_size = params.batch_size
        # self.train_loader = self.get_loader(adata_X, args_transformation)

        self.model = SCLE(self.dims, params.dropout).to(self.device)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=params.cl_lr)

        self.X_sim = X_sim.X    # single cell synthetic data
        self.X_label = X_label
        self.cls = Classification(self.dims[-1], X_label.shape[1]).to(self.device)
        self.cls_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.cls.parameters()), lr=params.cls_lr)
        self.criterion_rep = st_loss.WeightedSupConLoss(temperature=params.temperature)

    def pre_train(self):
        ''' pre-training contrastive model'''
        self.model.train()
        bar = Bar('SCLE model train without DEC: ', max=self.epochs)
        bar.check_tty = False
        losses = []
        idx = np.arange(len(self.X))
        for epoch in range(self.epochs):
            start_time = time.time()
            self.model.train()
            adjust_learning_rate(self.optimizer, epoch, self.params.cl_lr)
            np.random.shuffle(idx)
            loss = 0
            for pre_index in range(len(self.X) // self.batch_size + 1):
                c_idx = np.arange(pre_index * self.batch_size,
                                  min(len(self.X), (pre_index + 1) * self.batch_size))
                if len(c_idx) == 0:
                    continue
                c_idx = idx[c_idx]
                c_inp = self.X[c_idx]
                label_inp = self.labels[c_idx]
                A_inp = self.A.tocsc()[c_idx][:, c_idx].todense()
                label_inp = torch.tensor(label_inp)
                A_inp = torch.tensor(A_inp)
                if self.params.noise is None or self.params.noise == 0:
                    input1 = torch.tensor(c_inp, dtype=torch.float).to(self.device)
                    input2 = torch.tensor(c_inp, dtype=torch.float).to(self.device)
                else:
                    noise_vec = np.random.normal(loc=0, scale=self.params.noise, size=c_inp.shape)
                    input1 = torch.tensor(c_inp + noise_vec, dtype=torch.float).to(self.device)
                    noise_vec = np.random.normal(loc=0, scale=self.params.noise, size=c_inp.shape)
                    input2 = torch.tensor(c_inp + noise_vec, dtype=torch.float).to(self.device)
                anchors_output = self.model(input1)
                neighbors_output = self.model(input2)

                features = torch.cat(
                    [anchors_output.unsqueeze(1),
                     neighbors_output.unsqueeze(1)],
                    dim=1)
                # total_loss = self.criterion_rep(features)
                total_loss = self.criterion_rep(features, labels=label_inp, weights=A_inp)  # take neighbor as label
                loss += total_loss.item()

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
            end_time = time.time()
            batch_time = end_time - start_time
            bar_str = '{} / {} | Left time: {batch_time:.2f} mins| Loss: {loss:.4f}'
            bar.suffix = bar_str.format(epoch + 1, self.epochs,
                                        batch_time=batch_time * (self.epochs - epoch) / 60, loss=loss)
            bar.next()
        bar.finish()

    def save_model(self, save_model_file):
        torch.save({'state_dict': self.model.state_dict()}, save_model_file)
        print('Saving model to %s' % save_model_file)

    def load_model(self, save_model_file):
        saved_state_dict = torch.load(save_model_file)
        self.model.load_state_dict(saved_state_dict['state_dict'])
        print('Loading model from %s' % save_model_file)

    def process(self, input):
        self.model.eval()
        latent_z = self.model(torch.tensor(input, dtype=torch.float).to(self.device))
        latent_z = latent_z.data.cpu().numpy()
        return latent_z

    def deconvolution_fit(self, Y_label=None):
        '''training classification to deconvolute each spot'''
        self.pre_train()
        loss_l = []
        loss_v = []
        val_x = torch.tensor(self.process(self.X)).to(self.device)
        if Y_label is not None:
            val_y = Y_label
        train_x = torch.tensor(self.process(self.X_sim)).to(self.device)
        train_y = torch.tensor(self.X_label, dtype=torch.float).to(self.device)
        train_loss = torch.nn.KLDivLoss(reduction='batchmean')
        bar = Bar('Deconvolution: ', max=self.params.cls_epochs)
        bar.check_tty = False
        for ep in range(self.params.cls_epochs):
            start_time = time.time()
            self.cls.train()
            self.cls_optimizer.zero_grad()
            out = self.cls(train_x)
            out = F.log_softmax(out, dim=1)
            loss = train_loss(out, train_y)
            loss.backward()
            self.cls_optimizer.step()
            loss_l.append(loss)
            end_time = time.time()
            batch_time = end_time - start_time
            if Y_label is not None:
                self.cls.eval()
                with torch.no_grad():
                    v_loss = math.sqrt(
                        mean_squared_error(F.softmax(self.cls(val_x), dim=1).cpu().detach().numpy(), val_y))
                    loss_v.append(v_loss)
                bar_str = '{} / {} | Left time: {batch_time:.2f} mins| T Loss: {t_loss:.4f}| V Loss：{v_loss:.4f}'
                bar.suffix = bar_str.format(ep + 1, self.params.cls_epochs,
                                            batch_time=batch_time * (self.params.cls_epochs - ep) / 60,
                                            t_loss=loss.item(), v_loss=v_loss)
                bar.next()
            else:
                bar_str = '{} / {} | Left time: {batch_time:.2f} mins| Loss: {loss:.4f}'
                bar.suffix = bar_str.format(ep + 1, self.params.cls_epochs,
                                            batch_time=batch_time * (self.params.cls_epochs - ep) / 60, loss=loss.item())
                bar.next()

        bar.finish()
        x = range(0, self.params.cls_epochs)
        y1 = loss_l
        y2 = loss_v
        plt.title('loss:')

        if Y_label is not None:
            plt.plot(x, y2, label='v_loss')
        else:
            plt.plot(x, y1, label='t_loss')
        plt.xlabel('iteration times')
        plt.ylabel('loss')
        # plt.show()
        predict = self.cls(val_x)
        predict = F.softmax(predict).cpu().detach().numpy()
        return predict






