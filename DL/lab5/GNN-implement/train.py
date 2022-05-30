import torch
import torch.nn as nn
import torch.optim as optim
from data import *
from model import *
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch.autograd as autograd
import random
'''
def train(model, train_loader, optimizer, epochs, criterion, lr_scheduler,
          test_loader):
    metrics_best = (0, 0, 0)
    for epoch in range(epochs):
        model.train()
        for data, target in tqdm(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        lr_scheduler.step()
        print('Epoch {}/{} Loss: {:.4f} '.format(epoch, epochs, loss.item()))
        metrics = test(model, test_loader)
        if metrics[0] > metrics_best[0]:
            metrics_best = metrics
            torch.save(model.state_dict(), './checkpoint/model.pt')
    print('Best result:')
    print('\tMAE: {:.4f}'.format(metrics_best[0]), end=' ')
    print('MRE: {:.4f}'.format(metrics_best[1]))
'''


def train_GAN(model_G,
              model_D,
              data_loader,
              optimizer_G,
              optimizer_D,
              epochs,
              n_noise=2):

    adversarial_loss = torch.nn.BCELoss()

    for epoch in range(epochs):
        loss_Ds = []
        loss_Gs = []
        for i, data in enumerate(data_loader):
            data = data[0].cuda()

            # Adversarial ground truths
            valid = torch.Tensor(data.size(0), 1).fill_(1.0).cuda()
            fake = torch.Tensor(data.size(0), 1).fill_(0.0).cuda()

            # ---------------------
            #  Train Generator
            # ---------------------

            optimizer_G.zero_grad()

            # Sample noise as generator input
            noise = torch.rand([data.size(0), n_noise]).cuda()

            # Generate a batch of images
            gen_imgs = model_G(noise)

            # Loss measures generator's ability to fool the discriminator
            loss_G = adversarial_loss(model_D(gen_imgs), valid)

            loss_G.backward()
            loss_Gs.append(loss_G.item())
            optimizer_G.step()

            # -----------------
            #  Train Discriminator
            # -----------------

            optimizer_D.zero_grad()

            real_loss = adversarial_loss(model_D(data), valid)
            fake_loss = adversarial_loss(model_D(gen_imgs.detach()), fake)
            loss_D = (real_loss + fake_loss) / 2
            loss_Ds.append(loss_D.item())

            loss_D.backward()
            optimizer_D.step()

        print("[Epoch %d/%d] [D loss: %f] [G loss: %f]" %
              (epoch, epochs, np.average(loss_Ds), np.average(loss_Gs)))

        if epoch % 10 == 0:
            test(model_G, model_D, data_loader, n_noise)
            # save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)


def train_WGAN(model_G,
               model_D,
               data_loader,
               optimizer_G,
               optimizer_D,
               epochs,
               n_noise=2,
               clamp=0.01):
    for epoch in range(epochs):
        loss_Ds = []
        loss_Gs = []
        for i, data in enumerate(data_loader):

            data = data[0].cuda()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            noise = torch.rand([data.size(0), n_noise]).cuda()

            # Generate a batch of images
            fake_imgs = model_G(noise).detach()
            # Adversarial loss
            loss_D = -torch.mean(model_D(data)) + torch.mean(
                model_D(fake_imgs))
            loss_Ds.append(loss_D.item())

            loss_D.backward()
            optimizer_D.step()

            # Clip weights of model_D
            for p in model_D.parameters():
                p.data.clamp_(-clamp, clamp)

            # Train the generator every n_critic iterations
            if i % 2 == 0:

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Generate a batch of images
                gen_imgs = model_G(noise)
                # Adversarial loss
                loss_G = -torch.mean(model_D(gen_imgs))
                loss_Gs.append(loss_G.item())

                loss_G.backward()
                optimizer_G.step()

        print("[Epoch %d/%d] [D loss: %f] [G loss: %f]" %
              (epoch, epochs, np.average(loss_Ds), np.average(loss_Gs)))

        if epoch % 10 == 0:
            test(model_G, model_D, data_loader, n_noise)
            # save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = random.random()
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples +
                    ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.Tensor(real_samples.shape[0], 1).fill_(1.0).cuda()
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean()
    return gradient_penalty


def train_WGAN_GP(model_G,
                  model_D,
                  data_loader,
                  optimizer_G,
                  optimizer_D,
                  epochs,
                  n_noise=2,
                  clamp=0.01):

    lambda_gp = 10

    for epoch in range(epochs):
        loss_Ds = []
        loss_Gs = []
        for i, data in enumerate(data_loader):

            data = data[0].cuda()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            noise = torch.rand([data.size(0), n_noise]).cuda()

            # Generate a batch of images
            fake_imgs = model_G(noise)

            # Real images
            real_validity = model_D(data)
            # Fake images
            fake_validity = model_D(fake_imgs)
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(
                model_D, data.data, fake_imgs.data)

            # Adversarial loss
            loss_D = -torch.mean(real_validity) + torch.mean(
                fake_validity) + lambda_gp * gradient_penalty
            loss_Ds.append(loss_D.item())

            loss_D.backward()
            optimizer_D.step()

            # Train the generator every 2 iterations
            if i % 2 == 0:

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Generate a batch of images
                gen_imgs = model_G(noise)
                # Adversarial loss
                loss_G = -torch.mean(model_D(gen_imgs))
                loss_Gs.append(loss_G.item())

                loss_G.backward()
                optimizer_G.step()

        print("[Epoch %d/%d] [D loss: %f] [G loss: %f]" %
              (epoch, epochs, np.average(loss_Ds), np.average(loss_Gs)))

        if epoch % 10 == 0:
            test(model_G, model_D, data_loader, n_noise)
            # save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)


def draw_background(model_D, x_min, x_max, y_min, y_max):
    xline = np.linspace(x_min, x_max, 100)
    yline = np.linspace(y_min, y_max, 100)
    bg = np.array([(x, y) for x in xline for y in yline])
    color = model_D(torch.Tensor(bg).cuda())
    color = (color - color.min()) / (color.max() - color.min())
    # print(color.shape)
    cm = plt.cm.get_cmap('gray')
    sc = plt.scatter(bg[:, 0],
                     bg[:, 1],
                     c=np.squeeze(color.cpu().data),
                     cmap=cm)
    # 显示颜色等级
    cb = plt.colorbar(sc)
    return cb


def test(model_G, model_D, data_loader, n_noise=2):
    with torch.no_grad():
        plt.clf()
        t = torch.rand([len(data_loader), n_noise])
        t = t.cuda()
        t = model_G(t)
        t = t.transpose(0, 1).cpu().data.numpy()

        draw_background(model_D, -0.5, 1.5, 0, 1)
        plt.xlim(-0.5, 1.5)
        plt.ylim(0, 1)

        # data_loader.show_plt()
        plt.scatter(data_loader.dataset.tensors[0].T[0],
                    data_loader.dataset.tensors[0].T[1],
                    alpha=0.2,
                    c='r')
        plt.scatter(t[0], t[1], label='g', c='b', alpha=0.2)

        plt.legend()
        # plt.savefig('./picWGAN/epoch'+str(epoch)+'.jpg')
        plt.show()