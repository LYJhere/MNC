"""Hi, plz refer to
#5 (comment)
The settings on STL-10 are the same as CC: only PSA on unlabeled data, and PSA+PSL on labeled data;
see https://github.com/Yunfan-Li/Contrastive-Clustering/blob/main/train_STL10.py

There are a little changes in my codebase, you can take the code below as reference:
"""
import torch


def set_loader(self):
    opt = self.opt
    dataset_name = opt.dataset

    if dataset_name not in dataset_dict.keys():
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        self.logger.msg_str(f'Dataset does {dataset_name} not exist in dataset_dict,'
                            f' use default normalizations: mean {str(mean)}, std {str(std)}.')
    else:
        mean, std = dataset_dict[dataset_name]

    normalize = transforms.Normalize(mean=mean, std=std, inplace=True)

    train_transform = self.train_transform(normalize)
    self.logger.msg_str('set transforms...')
    self.logger.msg_str(train_transform)

    self.logger.msg_str('set train and unlabeled dataloaders...')
    train_loader, labels, train_sampler = self.build_dataloader(
        transform=train_transform,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=True,
        sampler=True,
        train=True)
    unlabeled_loader, _, unlabeled_sampler = self.build_dataloader(
        transform=train_transform,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=True,
        sampler=True,
        train=False)

    test_transform = []
    if 'imagenet' in dataset_name:
        test_transform += [transforms.Resize(256), transforms.CenterCrop(224)]
    test_transform += [
        transforms.Resize(opt.img_size),
        transforms.ToTensor(),
        normalize
    ]

    test_transform = transforms.Compose(test_transform)

    # self.logger.msg_str('set test dataloaders...')
    # test_loader = self.build_dataloader(test_transform, train=False, batch_size=opt.batch_size)[0]
    self.logger.msg_str('set memory dataloaders...')
    memory_loader = self.build_dataloader(test_transform, train=True, batch_size=opt.batch_size, sampler=True)[0]

    # self.test_loader = test_loader
    self.train_loader = train_loader
    self.memory_loader = memory_loader
    self.unlabeled_loader = unlabeled_loader
    self.train_sampler = train_sampler
    self.unlabeled_sampler = unlabeled_sampler

    self.iter_per_epoch = len(train_loader) + len(unlabeled_loader)
    self.num_classes = len(np.unique(labels))
    self.num_samples = len(labels)
    self.indices = torch.zeros(len(self.train_sampler), dtype=torch.long).cuda()
    self.num_cluster = self.num_classes if opt.num_cluster is None else opt.num_cluster
    self.psedo_labels = torch.zeros((self.num_samples,)).long().cuda()

    self.logger.msg_str('load {} images...'.format(self.num_samples))


def fit(self):
    opt = self.opt
    # training routine
    self.progress_bar = tqdm.tqdm(total=self.iter_per_epoch * opt.epochs, disable=(self.rank != 0))

    n_iter = self.iter_per_epoch * opt.resume_epoch + 1
    self.progress_bar.update(n_iter)
    max_iter = opt.epochs * self.iter_per_epoch

    while True:
        epoch = int(n_iter // self.iter_per_epoch + 1)
        self.train_sampler.set_epoch(epoch)
        self.unlabeled_sampler.set_epoch(epoch)

        for inputs in self.unlabeled_loader:
            inputs = convert_to_cuda(inputs)
            self.train_unlabeled(inputs, n_iter)
            self.progress_bar.refresh()
            self.progress_bar.update()
            n_iter += 1

        apply_kmeans = epoch % opt.reassign == 0
        if apply_kmeans:
            self.psedo_labeling(n_iter)

        self.indices.copy_(torch.Tensor(list(iter(self.train_sampler))))
        for inputs in self.train_loader:
            inputs = convert_to_cuda(inputs)
            self.adjust_learning_rate(n_iter)
            self.train(inputs, n_iter)
            self.progress_bar.refresh()
            self.progress_bar.update()
            n_iter += 1
        # if epoch % opt.save_freq == 0:
        # self.logger.checkpoints(int(epcoch))
        # self.test(n_iter)

        if n_iter > max_iter:
            break


def train(self, inputs, n_iter):
    opt = self.opt

    images, labels = inputs
    self.byol.train()

    im_q, im_k = images

    _start = ((n_iter - 1) % self.iter_per_epoch - len(self.unlabeled_loader)) * opt.batch_size
    indices = self.indices[_start: _start + opt.batch_size]
    psedo_labels = self.psedo_labels[indices]

    # compute loss
    contrastive_loss, cluster_loss_batch, q = self.byol(
        im_q, im_k, psedo_labels)

    loss = contrastive_loss
    if ((n_iter - 1) / self.iter_per_epoch) > opt.warmup_epochs:
        loss += cluster_loss_batch * opt.cluster_loss_weight

    self.optimizer.zero_grad()
    # SGD
    loss.backward()
    self.optimizer.step()

    with torch.no_grad():
        q_std = torch.std(q.detach(), dim=0).mean()

    outputs = [contrastive_loss, cluster_loss_batch, q_std]
    self.logger.msg(outputs, n_iter)


def train_unlabeled(self, inputs, n_iter):
    opt = self.opt

    images, labels = inputs
    self.byol.train()

    im_q, im_k = images

    # compute loss
    unlabeled_contrastive_loss, _, q = self.byol(im_q, im_k, None)

    self.optimizer.zero_grad()
    # SGD
    unlabeled_contrastive_loss.backward()
    self.optimizer.step()

    with torch.no_grad():
        unlabeled_q_std = torch.std(q.detach(), dim=0).mean()

    outputs = [unlabeled_contrastive_loss, unlabeled_q_std]
    self.logger.msg(outputs, n_iter)
