import torch
import config
import torchvision
from utils import *
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

def train_fn(loader , gen , disc , opt_gen , opt_disc ,  fixed_noise, loss_fn):
    
    for epoch in range(1, config.NUM_EPOCHS + 1):

        with tqdm(loader) as progress_bar:
            for real, _ in progress_bar:
                    
                    gen.train()
                    disc.train()
                    
                    real = real.to(config.DEVICE)
                    batch_size = real.shape[0]

                    noise = torch.randn(batch_size , config.Z_DIM , 1 , 1).to(config.DEVICE)

                    # Training Discriminator
                    fake = gen(noise)
                    disc_real = disc(real).reshape(-1)
                    loss_disc_real = loss_fn(disc_real, torch.ones_like(disc_real))
                    disc_fake = disc(fake.detach()).reshape(-1)
                    loss_disc_fake = loss_fn(disc_fake, torch.zeros_like(disc_fake))
                    loss_disc = (loss_disc_real + loss_disc_fake) / 2
                    
                    disc.zero_grad()
                    loss_disc.backward()
                    opt_disc.step()

                    # Train generator
                    output = disc(fake).reshape(-1)
                    loss_gen = loss_fn(output, torch.ones_like(output))

                    gen.zero_grad()
                    loss_gen.backward()
                    opt_gen.step()

                    # Display messege
                    progress_bar.set_description(f' - Epoch {epoch}/{config.NUM_EPOCHS} - ')
                    progress_bar.set_postfix(loss_G = loss_gen.item(), loss_D = loss_disc.item())
                    
        # Write the messege to Tensorboard
        writer.add_scalar("D_Loss/Losses", loss_disc , epoch)
        writer.add_scalar("G_Loss/Losses", loss_gen , epoch)
        
        if epoch % config.SAVE_IMG_FREQ == 0:
            with torch.no_grad():

                fake = gen(fixed_noise)
                # img_grid_real = torchvision.utils.make_grid(real[:32], normalize = True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize = True)
                plot_examples( "fake_ep" + str(epoch) + ".png" , "saved_samples/" , img_grid_fake)
                # plot_examples("Real_ep"+str(epoch)+ ".png" , "saved_samples/" , img_grid_real)
        
        if epoch % config.SAVE_CHECKPOINT_FREQ == 0:
            save_checkpoint(gen , opt_gen , filename = 'ep' + str(epoch) + '_' + config.CHECKPOINT_GEN, dir = config.CHECKPOINT_DIR)
            save_checkpoint(disc , opt_disc , filename = 'ep' + str(epoch) + '_' + config.CHECKPOINT_DISC, dir = config.CHECKPOINT_DIR)
            messegeDividingLine(f' - Checkpoint saved! - ')
    
def main():

    checkGPU(config)

    transforms = getTransform(config)
    dataloader = getDataLoader(config, transforms)

    generator, discriminator = getModel(config)
    opt_gen, opt_disc = getOptimizer(config, generator.parameters(), discriminator.parameters())
    loss_fn = getLossFunction()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, generator, opt_gen, config.G_LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC , discriminator , opt_disc , config.D_LEARNING_RATE)

    messegeDividingLine(" - Training starts -")

    fixed_noise = torch.randn(32 , config.Z_DIM , 1 , 1).to(config.DEVICE)
    train_fn(dataloader, generator, discriminator, opt_gen, opt_disc, fixed_noise, loss_fn)

        
if __name__  == "__main__":
    main()
